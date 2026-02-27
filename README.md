# LLM Model Selection for GRPO + LoRA

This repository studies online model selection methods for post training LLMs with GRPO+LoRA adapters. The pipeline trains multiple base models ,each with unique configuration, and uses a meta-selection strategy to choose which base model to train at each episode, then logs training and reward behavior for comparison across selection methods. For memory efficiency, each base model is identified with a unique LoRA adaptor and the pretrained model is shared.

## Folder Structure

```text


+├── modsel_GRPO_LoRA/                      # Main training pipeline
+│   ├── dataset.py
+│   ├── main.py
+│   ├── reward_funcs.py
+│   ├── utils.py
+└── model_selection_algorithms/            # Model Selection + Bandit implementations
+    ├── bandit_algs.py
+    ├── modsel_algs.py

 ```


## Acknowledgment

This codebase partially reuses GRPO implementation from Hugging Face TRL and Reasoning RL notebooks from Unsloth AI.
