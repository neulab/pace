#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

. .env

while getopts ":s:m:l:r:o:p:t:e:" opt; do
  case ${opt} in
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    # \? ) echo "Usage: cmd [-p] [-m] [-l] [-o] [-pp] [-tp]";;
  esac
done

MAX_TOKEN=8192
export OPENAI_API_KEY=${LLM_API_KEY}
uv run --isolated evalplus.evaluate \
    --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
    --dataset mbpp \
    --backend openai \
    --temperature 1.0 \
    --base-url ${LLM_API_URL} \
    --max_new_tokens ${MAX_TOKEN}
