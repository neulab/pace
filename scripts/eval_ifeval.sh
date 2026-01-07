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

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$(( $RANDOM_PORT ))}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

TASK_LIST=(
    ifeval
)

MAX_TOKEN=8192
for TASK in ${TASK_LIST[@]}
do
    uv run yeval \
        --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
        --task "${TASK}" \
        --include_path proxy_bench/tasks/ \
        --api_base ${LLM_API_URL} \
        --api_key ${LLM_API_KEY} \
        --run_name $MODEL/$TASK \
        --trust_remote_code \
        --max_rps 10 \
        --output_path data/eval_scores/ $OTHER_ARGS

done
