#!/bin/bash

#SBATCH --job-name=livecodebench
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --partition=cpu
#SBATCH -c 16
#SBATCH --time 1-23:59:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yueqis@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxy

export OPENAI_API_KEY=rpa_DA04C3HTUPF2WGF7D3GY36NBT4UOY6FFOZ7NUYYJ1rv9kq
export OPENAI_BASE_URL=https://api.runpod.ai/v2/c35bkyozx7erir/openai/v1
REASONING=""
MODEL=nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16
IDX=0

echo $MODEL$REASONING
echo $IDX

python -m lcb_runner.runner.main --model $MODEL$REASONING --scenario codegeneration --idx $IDX --n 1 --evaluate --release_version release_v6 --multiprocess 16
python -m lcb_runner.runner.main --model $MODEL$REASONING --scenario selfrepair --codegen_n 1 --idx $IDX --n 1 --evaluate --release_version release_v6 --multiprocess 16
python -m lcb_runner.runner.main --model $MODEL$REASONING --scenario testoutputprediction --idx $IDX --n 1 --evaluate --release_version release_v6 --multiprocess 16
python -m lcb_runner.runner.main --model $MODEL$REASONING --scenario codeexecution --idx $IDX --n 1 --cot_code_execution --evaluate --release_version release_v6 --multiprocess 16
