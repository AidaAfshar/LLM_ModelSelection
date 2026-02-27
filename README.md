# LLM Model Selection for GRPO + LoRA
 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![Unsloth](https://img.shields.io/pypi/v/unsloth?label=Unsloth)](https://pypi.org/project/unsloth/)



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

This codebase partially reuses GRPO implementation from Hugging Face TRL and Reasoning RL codebase from Unsloth AI.
