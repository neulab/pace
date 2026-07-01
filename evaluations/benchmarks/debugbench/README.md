# Repo of DebugBench (Forked and Modified for ProxyBench)

This repository contains the source code used to run experiments for ProxyBench experiments.

## Instructions for Setup

### Installing Dependencies
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions for uv.
```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install litellm
```

### LeetCode Tokens
Perform the follow steps to get tokens to programmatically submit code solutions to Leetcode:
- Go to leetcode.com and login. If you haven't verified your email (for new accounts), do that before next step.
- Make sure the URL on the webpage is exactly "https://leetcode.com/". Right click and press inspect. Go to Application -> Storage -> Cookies -> leetcode.com
- Copy the strings for the LEETCODE_SESSION and csrftoken and paste them in [keys.json](evaluation/keys.json).

> IMPORTANT NOTE: Do NOT logout of your leetcode account else the tokens will expire. To speed up evals, secure tokens from multiple Leetcode accounts. Our code will route requests to reduce rate limit errors arising from repeatedly using the same tokens.

### Running Evaluations

We use LiteLLM for querying OpenAI compatible endpoints for all our experiments. Before starting experiments, make sure to set the below environment variables:
```bash
export LITELLM_API_KEY="<your_api_key>"
export LITELLM_BASE_URL="<base_url>"
```

To run evaluations, use the following command:
```bash
python3 evaluation/debug.py \
--start_idx <0-based starting index in keys.json for the token you want to use> \
--cnt <use multiple tokens - start_idx, ..., start_idx+cnt-1 \
--model "<model_name>" \
--timeout <minimum time interval between querying the same token>
```

For example:
```bash
python3 evaluation/debug.py \
--start_idx 0 \
--cnt 3 \
--model "<model_name>" \
--timeout 60
```

---

## Original DebugBench README
<img src="figs/icon.png" style="width: 20vw; height: auto;" alt="icon"> 

### Overview

Implementation for paper DebugBench: Evaluating Debugging Capabilities of Large Language Models with datasets, prompts, model outputs.



### Benchmark

Please refer to the [Hugging Face Dataset](https://huggingface.co/datasets/Rtian/DebugBench) for the data source and evaluation script if you want to use the benchmark.

DebugBench is a Large Language Model (LLM) debugging benchmark introduced in the paper "DebugBench: Evaluating Debugging Capability of Large Language Models" [url]. We collect code snippets from the [LeetCode](https://leetcode.com/) community and implant bugs into source data with [GPT-4](https://openai.com/research/gpt-4). 

- It consists of 4,253 instances.
- It covers four major bug categories and 18 minor types.
- It includes C++, Java, and Python instances.
- It contains three difficulty levels: easy, medium, and hard.
- All the instances were released after June 2022.
- Please refer to the article [url] for more details.



### Repo Content

This repository contains the implementation for benchmark construction and evaluation.

- `benchmark` directory contains the 51 JSON shards of different languages and bug types of the benchmark. 
- `dataset_construction` directory contains the implementation for bug implantation to solution code via LLMs.

- `evaluation` directory contains the implementation for evaluating the debugging capabilities of LLMs with API. 
- `evalution_result` directory contains the model output of `gpt-4-0613` ,  `gpt-3.5-turbo-0613`  and `CodeLlama-34b-instruct` under different scenarios. 

More elements will be added to the repository soon.



### Citations

Please cite the paper and star the repo if you use DebugBench and find it helpful.

Feel free to contact trc20@mails.tsinghua.edu.cn or open an issue if you have any questions.

```latex
@misc{tian2024debugbench,
      title={DebugBench: Evaluating Debugging Capability of Large Language Models}, 
      author={Runchu Tian and Yining Ye and Yujia Qin and Xin Cong and Yankai Lin and Zhiyuan Liu and Maosong Sun},
      year={2024},
      eprint={2401.04621},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

