
from unsloth import FastLanguageModel
import torch
import argparse
import random
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--G", type=int, default=4)
parser.add_argument("--temp", type=float, default=1)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

lora_rank = args.rank
learning_rate = args.lr
num_generations = args.G
seed = args.seed
temperature = args.temp
max_seq_length = 1024 #2048 # Can increase for longer reasoning traces


# seeding
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

"""
 LoRA Info
"""

from peft.tuners.lora import LoraLayer

def print_lora_shapes(model):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            base = module.base_layer
            a = base.out_features
            b = base.in_features

            for adapter_name in module.lora_A.keys():
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                r = A.shape[0]

                print(f"\nLayer: {name}")
                print(f"  Base weight W: ({a}, {b})")
                print(f"  LoRA rank r = {r}")
                print(f"  A: ({r}, {b})")
                print(f"  B: ({a}, {r})")
                print(f"  Decomposition: W({a}×{b}) ≈ B({a}×{r}) @ A({r}×{b})")

print_lora_shapes(model)


"""Data Prep: OpenAI's GSM8K dataset
"""

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split = "train")

"""Let's look at the first row:"""

print(dataset[0]["question"])

print(dataset[0]["answer"])

"""We notice all answers like about have a ####, so we extract it:"""

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()
extract_hash_answer(dataset[0]["answer"])

"""We now create a system prompt which can be customized. We add 4 extra symbols for working out or thinking / reasoning sections and a final answer:"""

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt

"""Let's map the dataset! and see the first row:"""

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})
dataset[0]

"""We create a regex format to match the reasoning sections and answers:"""

import re

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

"""We verify it works:"""

match_format.search(
    "<start_working_out>Let me think!<end_working_out>"\
    "<SOLUTION>2</SOLUTION>",
)

"""We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:"""

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores

"""Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

We also remove possible commas for example as in 123,456
"""

match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores

"""Get the maximum prompt length so we don't accidentally truncate it!"""

max(dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
).map(lambda x: {"length" : len(x["tokens"])})["length"])



"""
Train the model

"""

# ---- Weights & Biases logging ----
import os
import wandb

# Set your project name here (or via environment variable WANDB_PROJECT)
os.environ.setdefault("WANDB_PROJECT", "llama3b-grpo-lora")

# run_name = f"grpo_lora__lr{learning_rate}__r{lora_rank}" #f"grpo_lora_r{lora_rank}"
run_name = f"llama3b__lr{learning_rate}__G{num_generations}__rank{lora_rank}__seed{seed}"
wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name=run_name,
    config={
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "learning_rate": learning_rate,
        "G":num_generations,
        "T":temperature,
        "seed":seed
    },
)




# ---- Now set up GRPO Trainer and all configurations! ----

max_prompt_length = 287 + 1 # + 1 just in case!

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = learning_rate,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "constant",  #"cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = num_generations, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    temperature = temperature,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 250,
    max_grad_norm = 1.0,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    seed = seed
)

# Update W&B config with training args
try:
    wandb.config.update(training_args.to_dict(), allow_val_change=True)
except Exception:
    # Fallback if to_dict is unavailable
    wandb.config.update({k: getattr(training_args, k) for k in dir(training_args) if not k.startswith('_')}, allow_val_change=True)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args = training_args,
    train_dataset = dataset,
)

train_output = trainer.train()






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


# # CPU RAM usage 
# print("\n===== RAM (if available) =====")
# try:
#     import psutil
#     vm = psutil.virtual_memory()
#     print(f"RAM total: {vm.total/1024**3:.2f} GB | used: {vm.used/1024**3:.2f} GB | percent: {vm.percent}%")
# except Exception as e:
#     print(f"psutil not available: {e}")



# """
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:
# """

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

# output

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

# output

# """Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

# <a name="Save"></a>
# ### Saving to float16 for VLLM

# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
# """

# # Merge to 16bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# # Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# # Just LoRA adapters
# if False:
#     model.save_pretrained("model")
#     tokenizer.save_pretrained("model")
# if False:
#     model.push_to_hub("hf/model", token = "")
#     tokenizer.push_to_hub("hf/model", token = "")

# """
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# """

# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model", tokenizer,)
# # Remember to go to https://huggingface.co/settings/tokens for a token!
# # And change hf to your username!
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# # Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# # Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# # Save to multiple GGUF options - much faster if you want multiple!
# if False:
#     model.push_to_hub_gguf(
#         "hf/model", # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = "",
#     )

