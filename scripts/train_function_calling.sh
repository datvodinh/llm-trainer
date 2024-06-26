while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --hf_username)
            hf_username="$2"
            shift
            shift ;;
        --hf_token)
            hf_token="$2"
            shift
            shift ;;
        --wandb_token)
            wandb_token="$2"
            shift
            shift ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$hf_username" || -z "$hf_token" || -z "$wandb_token" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: train_function_calling.sh --hf_username <username> --hf_token <token> --wandb_token <token>"
    exit 1
fi


python -m llm_trainer.function_calling \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --rank 16 \
    --max_steps 2000 \
    --num_train_epochs 1 \
    --learning_rate 0.00002 \
    --logging_steps 10 \
    --optim "adamw_8bit" \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --seed 3407 \
    --output_dir "llama-3-8b-funtion-calling-q4_k_m-gguf" \
    --max_seq_length 2048 \
    --hf_username "$hf_username" \
    --finetune_model_name "llama-3-8b-funtion-calling-q4_k_m-gguf" \
    --hf_token "$hf_token" \
    --wandb_token "$wandb_token" \
    --quantization_method "q4_k_m"