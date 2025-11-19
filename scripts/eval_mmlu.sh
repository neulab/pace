#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

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
  mmlu_abstract_algebra
  mmlu_anatomy
  mmlu_astronomy
  mmlu_auxiliary_train
  mmlu_business_ethics
  mmlu_clinical_knowledge
  mmlu_college_biology
  mmlu_college_chemistry
  mmlu_college_computer_science
  mmlu_college_mathematics
  mmlu_college_medicine
  mmlu_college_physics
  mmlu_computer_security
  mmlu_conceptual_physics
  mmlu_econometrics
  mmlu_electrical_engineering
  mmlu_elementary_mathematics
  mmlu_formal_logic
  mmlu_global_facts
  mmlu_high_school_biology
  mmlu_high_school_chemistry
  mmlu_high_school_computer_science
  mmlu_high_school_european_history
  mmlu_high_school_geography
  mmlu_high_school_government_and_politics
  mmlu_high_school_macroeconomics
  mmlu_high_school_mathematics
  mmlu_high_school_microeconomics
  mmlu_high_school_physics
  mmlu_high_school_psychology
  mmlu_high_school_statistics
  mmlu_high_school_us_history
  mmlu_high_school_world_history
  mmlu_human_aging
  mmlu_human_sexuality
  mmlu_international_law
  mmlu_jurisprudence
  mmlu_logical_fallacies
  mmlu_machine_learning
  mmlu_management
  mmlu_marketing
  mmlu_medical_genetics
  mmlu_miscellaneous
  mmlu_moral_disputes
  mmlu_moral_scenarios
  mmlu_nutrition
  mmlu_philosophy
  mmlu_prehistory
  mmlu_professional_accounting
  mmlu_professional_law
  mmlu_professional_medicine
  mmlu_professional_psychology
  mmlu_public_relations
  mmlu_security_studies
  mmlu_sociology
  mmlu_us_foreign_policy
  mmlu_virology
  mmlu_world_religions
)

MAX_TOKEN=8192
for TASK in ${TASK_LIST[@]}
do
    uv run yeval \
        --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
        --sample_args "n=2" \
        --task "${TASK}" \
        --include_path proxy_bench/tasks/ \
        --api_base ${LLM_API_URL} \
        --api_key ${LLM_API_KEY} \
        --run_name $MODEL/$TASK \
        --trust_remote_code \
        --output_path data/eval_scores/ $OTHER_ARGS
        
done
