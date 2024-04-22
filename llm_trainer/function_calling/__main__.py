import os
import torch
import wandb
import huggingface_hub
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
from llm_trainer.utils import get_parse_arguments

if __name__ == "__main__":
    # ARGRUMENTS
    args = get_parse_arguments()

    # LOGIN
    huggingface_hub.login(token=args.hf_token)
    if args.wandb_token is not None:
        wandb.login(key=args.wandb_token)
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    token_mapping = {
        "bos": "<|begin_of_text|>",
        "eos": "<|end_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "function": "function",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>",
    }

    def format_dataset_func(examples):
        conversation = examples['conversation']

        texts = [token_mapping["bos"]]
        for conv in conversation:
            texts.append(token_mapping[conv["role"]] + "\n")
            texts.append(conv["content"] + "\n")

        texts.append(token_mapping["eos"])
        return {
            "text": "".join(texts),
        }

    dataset = load_dataset("dinhdat1110/glaive-function-calling-v2-cleaned", split="train")
    dataset = dataset.map(format_dataset_func)

    # TRAINER
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="wandb" if args.wandb_token is not None else None,
        ),
    )

    trainer.train()

    model.push_to_hub_gguf(
        f"{args.hf_username}/{args.finetune_model_name}",
        tokenizer,
        quantization_method="q4_k_m",
        token=args.hf_token,
    )
