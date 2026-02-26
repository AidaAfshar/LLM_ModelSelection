import copy
import torch
from trl import GRPOConfig, GRPOTrainer
import numpy as np
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from argparse import Namespace
from modsel_GRPO_LoRA.reward_funcs import match_format_exactly, match_format_approximately, check_answer, check_numbers, MAX_REWARD
from peft import get_peft_model_state_dict, set_peft_model_state_dict



class GRPORewardCallback(TrainerCallback):
    def __init__(self):
        self.reward_vals = []
        self.reward_std_vals = []
        self.component_rewards = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Main reward
        if "reward" in logs:
            self.reward_vals.append(float(logs["reward"]))

        if "reward_std" in logs:
            self.reward_std_vals.append(float(logs["reward_std"]))

        # Per-reward-function metrics
        for k, v in logs.items():
            if k.startswith("rewards/") and isinstance(v, (int, float)):
                if k not in self.component_rewards:
                    self.component_rewards[k] = []
                self.component_rewards[k].append(float(v))

    def summarize(self):
        out = {}

        if self.reward_vals:
            out["episode_reward_mean"] = float(np.mean(self.reward_vals))
            out["episode_reward_last"] = float(self.reward_vals[-1])
            out["episode_reward_std_mean"] = float(np.mean(self.reward_std_vals)) if self.reward_std_vals else None
            out["episode_reward_trajectory"] = self.reward_vals  # optional

        # Aggregate reward components
        for k, vals in self.component_rewards.items():
            out[f"{k}/mean"] = float(np.mean(vals))
            out[f"{k}/last"] = float(vals[-1])

        return out



def only_active_adapter_trainable(model, active_name: str):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if ("lora_" in n) and (active_name in n):
            p.requires_grad = True


def cpuify_state(sd):
    # move tensors to CPU
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in sd.items()}


def load_adapter_i(model, lora_states, i):
    DEFAULT_ADAPTER = next(iter(model.peft_config.keys()))
    set_peft_model_state_dict(model, lora_states[i], adapter_name=DEFAULT_ADAPTER)  # adapter_name None usually targets default

def save_adapter_i(model, lora_states, i):
    DEFAULT_ADAPTER = next(iter(model.peft_config.keys()))
    lora_states[i] = cpuify_state(get_peft_model_state_dict(model, adapter_name=DEFAULT_ADAPTER))
    return lora_states


def make_grpo_args(overrides: dict, args: Namespace):
    default_config = dict(
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_ratio=0.1,
        max_steps= args.H,
        lr_scheduler_type="constant",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.G,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_seq_length - args.max_prompt_length,
        temperature=args.temp,
        report_to= "none", #"wandb",
        output_dir="outputs",
        seed=args.seed,
        save_steps=0,
    )
    default_config.update(overrides)
    return GRPOConfig(**default_config)


def to_cpu_state_dict(state_dict):
    if isinstance(state_dict, dict):
        return {k: to_cpu_state_dict(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [to_cpu_state_dict(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.detach().cpu()
    else:
        return state_dict



def train_episode(model, tokenizer, dataset, lora_states, base_index: int, cfg_overrides: dict, 
                                opt_state_cpu, sched_state_cpu, trainer_state, args):
    cfg_overrides = cfg_overrides or {}
    load_adapter_i(model, lora_states, base_index)
    training_args = make_grpo_args(cfg_overrides, args)
    reward_cb = GRPORewardCallback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args=training_args,
        train_dataset=dataset,
        callbacks=[reward_cb],
    )
    
    trainer.create_optimizer_and_scheduler(num_training_steps=args.H)
    
    # (optional) restore optimizer state per adapter i
    if base_index in opt_state_cpu:
        trainer.optimizer.load_state_dict(opt_state_cpu[base_index])
    if base_index in sched_state_cpu and trainer.lr_scheduler is not None:
        trainer.lr_scheduler.load_state_dict(sched_state_cpu[base_index])
    
    trainer.train()

    
    # Save optimizer state per adapter i (CPU)
    opt_state_cpu[base_index] = to_cpu_state_dict(trainer.optimizer.state_dict())
    if trainer.lr_scheduler is not None:
        sched_state_cpu[base_index] = to_cpu_state_dict(trainer.lr_scheduler.state_dict())

    # Pull updated LoRA weights out of the single slot and store as adapter i
    save_adapter_i(model, lora_states, base_index)


    # Extract reward summary BEFORE cleanup
    episode_reward_summary = reward_cb.summarize()
    
    # del trainer
    
    torch.cuda.empty_cache()
    
    return episode_reward_summary





