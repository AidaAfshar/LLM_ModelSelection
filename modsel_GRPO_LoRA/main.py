import os
import random
import numpy

# Set these before importing torch/unsloth so CUDA/tokenizer backends pick them up.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unsloth import FastLanguageModel
import torch
from torch.utils.tensorboard import SummaryWriter
from modsel_GRPO_LoRA.dataset import prep_dataset
from modsel_GRPO_LoRA.reward_funcs import match_format_exactly, match_format_approximately, check_answer, check_numbers, MAX_REWARD
from modsel_GRPO_LoRA.utils import train_episode, cpuify_state, load_adapter_i, save_adapter_i
import argparse
import copy
from peft import LoraConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from model_selection_algorithms.modsel_algs import PerfectBalancing, BD3RB, BalancingClassic, BalancingHyperparamDoublingDataDriven, CorralHyperparam, EXP3Hyperparam, UCBHyperparam
import wandb




# ---- Parse Arguements ----
parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=5)
parser.add_argument("--modsel_alg", type=str, default="Perfect")
parser.add_argument("--perfect_mode", type=str, default="pessimistic")
parser.add_argument("--target_hparam", type=str, default="learning_rate")
parser.add_argument("--num_episodes", type=int, default=300)
parser.add_argument("--H", type=int, default=10)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--G", type=int, default=4)
parser.add_argument("--temp", type=float, default=1)  
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--max_prompt_length", type=int, default=288)   # 287 + 1 just in case
parser.add_argument("--d_min", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic_mode", action="store_true",
                    help="Prefer deterministic kernels/settings over maximum throughput without changing GRPO sampling hyperparameters.")

args = parser.parse_args()

M = args.M   #number of base learners
modsel_alg = args.modsel_alg
perfect_mode = args.perfect_mode
target_hparam = args.target_hparam

num_episodes = args.num_episodes
H = args.H
lora_rank = args.rank
learning_rate = args.lr
num_generations = args.G
temperature = args.temp
max_seq_length = args.max_seq_length
max_prompt_length = args.max_prompt_length
seed = args.seed

MAX_REWARD = 10
MIN_REWARD = -4


base_learning_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
if M == 1:
    base_learning_rates = [learning_rate]
    num_episodes = 300


# reproducibility / deterministic setup
os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

# ---- Load pretrained model and tokenizer ----
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = not args.deterministic_mode, # vLLM path is faster but less reproducible
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)


# ---- Initiate LoRA base adaptors ----
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = seed,
)


# Initialize M adapter snapshots 
base_lora = cpuify_state(get_peft_model_state_dict(model))
lora_states = [copy.deepcopy(base_lora) for _ in range(M)]



# ---- Initate Selector ----
selector = None
if modsel_alg == "Perfect":
    selector = PerfectBalancing(M, r_star=MAX_REWARD, mode=perfect_mode)
    # selector = PerfectBalancing(M, r_star=MAX_REWARD, mode="greedy")
    print(f"{perfect_mode} Perfect Balancing!")
elif modsel_alg == "D3RB" :
    dmin = 1 #args.d_min
    selector = BalancingHyperparamDoublingDataDriven(M, dmin = dmin)
elif modsel_alg == "BD3RB" :
    dmin = 1 #args.d_min
    selector = BD3RB(M, r_star = MAX_REWARD, dmin = dmin)
elif modsel_alg == "ED2RB" :
    dmin = 1 #args.d_min
    selector = BalancingHyperparamDoublingDataDriven(M, dmin = dmin, empirical = True)
elif modsel_alg == "Corral":
    selector = CorralHyperparam(M, eta = 1/num_episodes, T=num_episodes)
elif modsel_alg == "Exp3":
    selector = EXP3Hyperparam(M)
elif modsel_alg == "UCB": 
    selector = UCBHyperparam(M, confidence_radius=1)
elif modsel_alg == "Classic":
    putative_bounds_multipliers = [1]*M
    selector = BalancingClassic(M, putative_bounds_multipliers)


# ---- Prep dataset ----
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

dataset = prep_dataset("GSM8K", reasoning_start, reasoning_end, solution_start, solution_end)


"""Get the maximum prompt length so we don't accidentally truncate it!"""
max(dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
).map(lambda x: {"length" : len(x["tokens"])})["length"])


# ---- Setup Weights & Biases----
import os
import wandb

# Set your project name here (or via environment variable WANDB_PROJECT)
os.environ.setdefault("WANDB_PROJECT", "Modsel_GRPO_LoRA")

run_name = f"llama3b__{modsel_alg}__{target_hparam}__M{M}__G{num_generations}__T{temperature}__rank{lora_rank}__seed{seed}"
if modsel_alg=="Perfect":
    run_name = f"llama3b__{modsel_alg}__{target_hparam}__M{M}__G{num_generations}__T{temperature}__rank{lora_rank}__seed{seed}"
if M==1:
    run_name = f"llama3b__{modsel_alg}__lr{base_learning_rates[0]}__M{M}__G{num_generations}__T{temperature}__rank{lora_rank}__seed{seed}"
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name=run_name,
    config={
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "learning_rate": learning_rate,
        "G":num_generations,
        "T":temperature,
        "seed":seed,
        "target_hparam": target_hparam,
        "M": M,
        "modsel_alg": modsel_alg,
        "horizon": H,
        "num_episodes": num_episodes
    },
    sync_tensorboard=True
)
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
RUN_ID = run.id



# ---- Training ----
# reward_history = {}
opt_state_cpu = {}
sched_state_cpu = {}
trainer_state = {}
selected_bases = []
for ep in range(num_episodes):
    # Select Base Models
    base_index = selector.sample_base_index()
    selected_bases.append(base_index)
    cfg_overrides = {
        "learning_rate": base_learning_rates[base_index]
    }
    
    # Train Base Model
    stats = train_episode(model, tokenizer, dataset, lora_states, base_index, cfg_overrides, 
                            opt_state_cpu, sched_state_cpu, trainer_state, args)
   
    
    print(
        f"[EP {ep:04d}] {base_index} | "
        f"reward_mean={stats.get('episode_reward_mean', float('nan')):.3f} | "
    )
    print("*** stats: ", stats)
    
    episode_reward_mean = stats.get('episode_reward_mean', float('nan'))
    normalized_episodic_reward = (episode_reward_mean - MIN_REWARD)/(MAX_REWARD - MIN_REWARD)
    # Update Selector
    if modsel_alg=="Perfect" or modsel_alg=="BD3RB" or modsel_alg=="D3RB" or modsel_alg=="ED2RB":
        selector.update_distribution(base_index, episode_reward_mean)
    else :  #other modsels require normalized reward
        selector.update_distribution(base_index, normalized_episodic_reward)


    if wandb.run is None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Modsel_GRPO_LoRA"),
            id=RUN_ID,
            resume="allow",
            reinit=True,
        )

    wandb.log(
    {
        "modelselection/metalearner_episodic_reward": episode_reward_mean,
        f"modelselection/base_{base_index}_episodic_reward": episode_reward_mean,
        "modelselection/selected_base_learner": base_index,
        "modelselection/learning_rate": base_learning_rates[base_index],
    },
    step=ep,
    )






# # ---- System Info ----
# print("\n===== System Info =====")
# print(f"Platform: {platform.platform()}")
# try:
#     import torch
#     print(f"PyTorch: {torch.__version__}")
#     if torch.cuda.is_available():
#         dev = torch.cuda.current_device()
#         props = torch.cuda.get_device_properties(dev)
#         print(f"GPU: {torch.cuda.get_device_name(dev)}")
#         print(f"Compute capability: {props.major}.{props.minor}")
#         print(f"Total VRAM: {props.total_memory/1024**3:.2f} GB")
#         print(f"CUDA: {torch.version.cuda}")
#         # Memory stats (current + peak)
#         allocated = torch.cuda.memory_allocated(dev) / 1024**3
#         reserved  = torch.cuda.memory_reserved(dev)  / 1024**3
#         max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
#         max_res   = torch.cuda.max_memory_reserved(dev)  / 1024**3
#         print(f"VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
#         print(f"VRAM peak allocated: {max_alloc:.2f} GB | peak reserved: {max_res:.2f} GB")
#     else:
#         print("CUDA not available.")
# except Exception as e:
#     print(f"Could not query torch/cuda info: {e}")


# """ CPU RAM usage """ 
# print("\n===== RAM (if available) =====")
# try:
#     import psutil
#     vm = psutil.virtual_memory()
#     print(f"RAM total: {vm.total/1024**3:.2f} GB | used: {vm.used/1024**3:.2f} GB | percent: {vm.percent}%")
# except Exception as e:
#     print(f"psutil not available: {e}")






# ---- Inference ----
# """Now let's try the model we just trained! First, let's first try the model without any GRPO trained:"""

# text = tokenizer.apply_chat_template([
#     {"role": "user", "content": "What is the sqrt of 101?"},
# ], tokenize = False, add_generation_prompt = True)

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 0.8,
#     top_p = 0.95,
#     max_tokens = 1024,
# )
# output = model.fast_generate(
#     [text],
#     sampling_params = sampling_params,
#     lora_request = None,
# )[0].outputs[0].text

# print(output)

# """And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

# model.save_lora(f"grpo_saved_lora_{lora_rank}")

# """Verify LoRA is actually trained!"""

# from safetensors import safe_open

# tensors = {}
# with safe_open(f"grpo_saved_lora_{lora_rank}/adapter_model.safetensors", framework = "pt") as f:
#     # Verify both A and B are non zero
#     for key in f.keys():
#         tensor = f.get_tensor(key)
#         n_zeros = (tensor == 0).sum() / tensor.numel()
#         assert(n_zeros.item() != tensor.numel())

# """Now we load the LoRA and test:"""

# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user",   "content": "What is the sqrt of 101?"},
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt = True, # Must add for generation
#     tokenize = False,
# )
# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 0.8,
#     top_p = 0.95,
#     max_tokens = 1024,
# )
# output = model.fast_generate(
#     text,
#     sampling_params = sampling_params,
#     lora_request = model.load_lora(f"grpo_saved_lora_{lora_rank}"),
# )[0].outputs[0].text

# print(output)
