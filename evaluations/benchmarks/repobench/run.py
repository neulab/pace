import os
import fire
import json
import litellm
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict, Dataset
import pandas as pd
from data.utils import construct_prompt

# get first line that is not a comment
def get_first_line_not_comment(code:str, language:str="python"):
    """
    This function gets the first line of code that is not a comment.

    Args:
    code: Str, the code

    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """

    # check if the language is valid
    assert language in ["python", "java"], "language must be one of [python, java]"


    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    in_multiline_comment = False

    if language == "python":
        # Remove ```python and ``` if they exist
        if lines[0].strip().startswith("```python"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('#'):
                continue
            # if the line is not a comment, then return the line
            return line
        
    elif language == "java":
        for line in lines:
            # if the line is empty, then skip
            if not line.strip():
                continue
            # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
            if not in_multiline_comment and line.strip().startswith('/*'):
                in_multiline_comment = True
                continue
            # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
            if in_multiline_comment and line.strip().endswith('*/'):
                in_multiline_comment = False
                continue
            # if the line is in a multiline comment, then skip
            if in_multiline_comment:
                continue
            # if the line is a single line comment, then skip
            if line.strip().startswith('//'):
                continue
            # if the line is not a comment, then return the line
            return line


    # if we cannot find a line that is not a comment, then return the first line
    return lines[0]

def filter_dataset_by_date_range(dataset: DatasetDict, start_date: str, end_date: str) -> DatasetDict:
    """
    Filters a Huggingface dataset by a specific date range.
    
    Parameters:
    dataset (DatasetDict): The input dataset with subsets containing a 'created_at' column.
    start_date (str): The start date in the format 'YYYY-MM-DD'.
    end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
    DatasetDict: The filtered dataset.
    """
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    filtered_dataset_dict = {}
    
    for subset_name in dataset.keys():
        df = pd.DataFrame(dataset[subset_name])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Filter the DataFrame
        mask = (df['created_at'] >= start_date) & (df['created_at'] <= end_date)
        filtered_df = df[mask]
        
        # Convert back to Huggingface Dataset
        filtered_dataset_dict[subset_name] = Dataset.from_pandas(filtered_df)
    
    return DatasetDict(filtered_dataset_dict)

def filter_dataset_by_levels(dataset: DatasetDict, levels: list) -> DatasetDict:
    """
    Filters a Huggingface dataset by specific levels.
    
    Parameters:
    dataset (DatasetDict): The input dataset with subsets containing a 'level' column.
    levels (list): The list of levels to filter by.
    
    Returns:
    DatasetDict: The filtered dataset.
    """
    filtered_dataset_dict = {}

    for subset_name in dataset.keys():
        # Filter the subset directly using the 'filter' method
        filtered_subset = dataset[subset_name].filter(lambda example: example['level'] in levels)
        filtered_dataset_dict[subset_name] = filtered_subset
    
    return DatasetDict(filtered_dataset_dict)

def main(
    model_name: str = "litellm_proxy/neulab/deepseek-coder-1.3b-base",
    dataset_name: str = "tianyang/repobench_python_v1.1",
    start_date: str = "2023-12-01", # YYYY-MM-DD
    end_date: str = "2023-12-31", # YYYY-MM-DD
    max_token_nums: int = None,  # max token number for the prompt (unused without local tokenizer)
    levels = ["2k", "4k", "8k", "12k", "16k"], # 24k, 32k, 64k and 128k are also available, but the number of them is limited
    language: str = "python",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 128, # max number of tokens to generate
    res_dir: str = "./results",
    debug: bool = False,
    ):

    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL", "https://cmu.litellm.ai")

    # Load the dataset
    dataset = load_dataset(dataset_name) #, ignore_verifications=True)

    # Filter the dataset by date range
    dataset = filter_dataset_by_date_range(dataset, start_date, end_date)

    # Filter the dataset by levels
    dataset = filter_dataset_by_levels(dataset, levels)

    # Create the save directory
    save_dir = f"{res_dir}/{model_name.split('/')[-1]}-{language}-{levels}"
    os.makedirs(save_dir, exist_ok=True)

    for subset, data in dataset.items():
        for i in tqdm(range(len(data)), desc=f"Generating {subset}"):
            prompt = construct_prompt(data[i], tokenizer=None, max_token_nums=max_token_nums, language=language)
            result = ""
            try:
                response = litellm.completion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a code completion assistant. Continue the provided code snippet directly and exactly from where it ends. Output only the next line of code, nothing else. No explanation, no markdown, no preamble."},
                        {"role": "user", "content": prompt},
                    ],
                    # temperature=temperature,
                    # top_p=top_p,
                    max_tokens=max_new_tokens,
                    # stop=["\n"],
                )
                # result = response
                # print("response", response)
                result = response.choices[0].message.content or ""
                if debug:
                    print(f"\n[DEBUG] item={i} subset={subset}\n[PROMPT]\n{prompt}\n[RESPONSE]\n{result}\n")
            except litellm.APIError as e:
                print(f"LiteLLM API error on item {i} ({subset}): {e}")
                continue
            except litellm.Timeout as e:
                print(f"LiteLLM timeout on item {i} ({subset}): {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on item {i} ({subset}): {e}")
                continue

            postprocess_result = get_first_line_not_comment(result, language=language)

            with open(f"{save_dir}/{subset}.jsonl", "a") as f_out:
                f_out.write(json.dumps({"idx": i, "level": data[i]["level"], "pred": postprocess_result, "gt": data[i]["next_line"]}) + "\n")

if __name__ == "__main__":
    fire.Fire(main)
