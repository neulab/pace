#!/bin/bash
#SBATCH --job-name=pxy
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1

. .env


# export LITELLM_API_KEY=
# export LITELLM_BASE_URL=https://openrouter.ai/api/v1
# model=qwen/qwen3-coder
# model=nvidia/nemotron-3-nano-30b-a3b

uv run --isolated run.py \
    --model_name $1 \
    --dataset_name "tianyang/repobench_python_v1.1" \
    --start_date "2023-12-01" \
    --end_date "2023-12-31" \
    --language "python" \
    --max_token_nums 15800 \
    --levels "8k" \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_new_tokens 4096
    # --debug \
