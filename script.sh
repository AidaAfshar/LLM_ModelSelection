#!/bin/bash -l


#$ -l gpus=1
#$ -l gpu_type=H200|A100
#$ -j y


module load miniconda
conda activate modselfm
module load cuda

wandb login 59559bb3b2e76b1a9283871e5f50da20bdcf6ac2

python llama3b_grpo_lora.py  \
  ${RANK:+--r $RANK} \
  ${LR:+--lr $LR} \
  ${G:+--G $G} \
  ${T:+--temp $T} \
  ${SEED:+--seed $SEED} 
