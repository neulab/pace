# Proxy Bench

```
for TASK in mmlu ifeval mbpp
do
    sbatch scripts/eval_${TASK}.sh -m neulab/qwen3-235b-a22b
    sbatch scripts/eval_${TASK}.sh -m neulab/qwen3-coder-480b-a35b-instruct
    sbatch scripts/eval_${TASK}.sh -m neulab/gpt-oss-120b
    sbatch scripts/eval_${TASK}.sh -m azure/gpt-4o
done
```
