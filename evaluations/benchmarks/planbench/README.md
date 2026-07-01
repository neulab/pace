# LLMs and Planning

This repo has the code for four papers:
1. The code in 'plan-bench' subdirectory belongs to the paper ["PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change"](https://arxiv.org/abs/2206.10498)
2. The code in 'llm_planning_analysis' subdirectory belongs to the paper ["On the Planning Abilities of Large Language Models--A Critical Investigation"](https://arxiv.org/abs/2305.15771)
3. The code in 'llm_planning_analysis/back_prompting_parallel.py' consists of an implementation of the [LLM-Modulo framework](https://openreview.net/forum?id=Th8JPEmH4z)
4. **NEW**: 'llm_planning_analysis' subdirectory also contains the code for the paper ["A Systematic Evaluation of the Planning and Scheduling Abilities of the Reasoning Model o1"](https://openreview.net/forum?id=FkKBxp0FhR)


# PlanBench Static Test Set Leaderboard

The leaderboard below shows the performance of the models on the PlanBench static test set with zero-shot prompting. Check out llm_planning_analysis/results/ folder for the detailed files. For Blocksworld Hard, the results are included in results/backprompting/ folder.

| Model Name | Model Type | Blocksworld - NL - 600 instances | Mystery Blocksworld - NL - 600 instances | Randomized Mystery Blocksworld - NL - 600 instances | Blocksworld Hard - PDDL - 110 instances |
|------------|------------|----------------------------------|----------------------------------------|--------------------------------------------------|----------------------------------------|
| Deepseek R1 | LRM | 99.1% | 43.3% | 25.8% | 53.6% |
| o1-preview | LRM | 97.8% | 52.8% | 37.3% | 23.65% |
| o1-mini | LRM | 56.6% | 19.1% | 3.5% | 10% |
| Claude-3.5 Sonnet | LLM | 54.8% | 0% | - | - |
| GPT-4o | LLM | 35.5% | 0% | - | - |
| LLaMA-3.1 405B | LLM | 62.6% | 0.8% | - | - |
| Claude 3 Opus | LLM | 59.3% | 0% | - | - |
| LLaMA-3 70B | LLM | 34.16% | 0% | - | - |
| GPT-4 | LLM | 34.6% | 0% | - | - |
| Gemini 1.5 Pro | LLM | 23.8% | - | - | - |

> Note: LLM = Large Language Model, LRM = Language Reasoning Model, NL = Natural Language Prompting, PDDL = Planning Domain Definition Language Prompting
## Submitting to the Leaderboard

Kindly submit results of any new models by submitting a pull request with the result file and the leaderboard will be updated.

## Citation(s)

PlanBench - _NeurIPS 2023 Datasets and Benchmarks Track_:
```
@article{valmeekam2023planbench,
  title={Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change},
  author={Valmeekam, Karthik and Marquez, Matthew and Olmo, Alberto and Sreedharan, Sarath and Kambhampati, Subbarao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={38975--38987},
  year={2023}
}
```

On the Planning Abilities of Large Language Models - _NeurIPS 2023 Spotlight_:
```
@article{valmeekam2023planning,
  title={On the planning abilities of large language models-a critical investigation},
  author={Valmeekam, Karthik and Marquez, Matthew and Sreedharan, Sarath and Kambhampati, Subbarao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={75993--76005},
  year={2023}
}
```

A Systematic Evaluation of the Planning and Scheduling Abilities of the Reasoning Model o1 - _TMLR_:
```
@article{valmeekam2025a,
title={A Systematic Evaluation of the Planning and Scheduling Abilities of the Reasoning Model o1},
author={Karthik Valmeekam and Kaya Stechly and Atharva Gundawar and Subbarao Kambhampati},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=FkKBxp0FhR},
note={}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=karthikv792/LLMs-Planning&type=Date)](https://www.star-history.com/#karthikv792/LLMs-Planning&Date)

