# Finetuning script for Llama 3 in different tasks, using Unsloth and PEFT

## Spec

- ~ 9Gb VRAM (Batch size 2, Accumulate Grad 4)

## Setup

### Colab and Local

```bash
source scripts/setup_colab.sh
```

### Kaggle

```bash
source scripts/setup_kaggle.sh
```

## Train

### Function Calling

```bash
source scripts/train_function_calling.sh \
--hf_username dinhdat1110 \
--hf_token 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 \
--wandb_token 844fc0e4bcb3ee33a64c04b9ba845966de80180e
```

### Instruction

- Coming Soon

## Todo

- [ ] Instruction Fine-tuning.
- [ ] DPO Trainer.
- [ ] Custom dataset for different task.
