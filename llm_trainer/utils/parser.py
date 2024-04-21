import argparse


def get_parse_arguments():
    parser = argparse.ArgumentParser(description="Training Arguments")

    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit",
                        help="Model name.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_ratio", type=int, default=1,
                        help="Warmup ratio.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every X updates steps.")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay if we apply some.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="Type of learning rate scheduler to use.")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model checkpoints and outputs.")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Maximum sequence length.")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Data type.")
    parser.add_argument("--load_in_4bit", action='store_true',
                        help="Load data in 4-bit format.")
    parser.add_argument("--hf_username", type=str, default=None,
                        help="Hugging Face username.")
    parser.add_argument("--finetune_model_name", type=str, default="llama-3-8b-instruct-funtion-calling-v1",
                        help="Model name to push to the Hugging Face Hub.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token.")
    parser.add_argument("--wandb_token", type=str, default=None,
                        help="Wandb API token.")

    args = parser.parse_args()
    return args
