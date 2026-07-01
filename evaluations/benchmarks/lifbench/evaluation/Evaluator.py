import itertools
import os
import glob

import pandas as pd
from ARS_Scores import ScoreTools
from tqdm import tqdm


from EvaluateFunc import *


class Evaluator():
    def __init__(self, data_dir='123123'):
        print(data_dir)
        self.base_path = data_dir
        self.func_map = {
            "onedoc-repeat": judge_onedoc_repeat,
            "onedoc-qa": judge_onedoc_qa,
            "onedoc-extract": judge_onedoc_extract,
            "list-single_query_id": judge_label_equal_output,
            "list-multi_query_id": judge_labels_equal_outputs,
            "list-offset_query_id": judge_label_equal_output,
            "list-offset_query_element": judge_label_equal_output,
            "list-blur_offset_query_id": judge_list_input_blur_offset_query,
            "list-blur_offset_query_element": judge_list_input_blur_offset_query,
            "multidoc-batch_label": judge_multidoc_batch_label,
            "multidoc-find_dup_text": judge_multidoc_find_dup_text,
            }

        self.model_list = os.listdir(os.path.join(self.base_path, 'outputs'))
        self.task_list = os.listdir(os.path.join(self.base_path, 'prompts'))

    def get_evaluate_func(self, output_file: str):
        assert '.json' in output_file
        task_name = os.path.split(output_file)[-1].split('.json')[0]
        assert task_name in self.func_map, f"Error: task {task_name} not in func_map"
        return self.func_map[task_name]

    def evaluate(self, model, task, append_mode=False):  # output_dict keys: prompt label parma length ins_id output
        '''
        must have "output", "params",
        optional: prompt
        return: scoreDict
        '''
        # process for task input
        task = task.replace('.json', '')
        # get the output_file from the model and the task
        output_file = f'{self.base_path}/outputs/{model}/{task}.json'
        output_file = glob.glob(output_file)
        if len(output_file) < 1:
            print(f'Warning: no output file found for model {model}, task {task}')
            return None
        output_file = output_file[0]
        with open(output_file, 'r') as f:
            outputs_list = json.load(f)

        evaluate_func = self.get_evaluate_func(output_file)
        for i, d in tqdm(enumerate(outputs_list), total=len(outputs_list), desc=f'{model}-{task}'):
            if 'output' not in d: continue
            if append_mode and 'score_dict' in d and isinstance(d['score_dict'], dict): continue
            d['score_dict'] = evaluate_func(d)
        with open(output_file, 'w') as f:
            json.dump(outputs_list, f, indent=2)

    def evaluate_all(self, models=[], tasks=[], append_mode=False):
        assert isinstance(models, list) and isinstance(tasks, list), f"models:{models}\n tasks:{tasks}"
        models = self.model_list if not models else models
        tasks = self.task_list if not tasks else tasks
        tasks = [t.replace('.json', '') for t in tasks]
        print(f'models: {models}')
        print(f'tasks: {tasks}')
        for model in models:
            assert model in self.model_list, f'Error: Model {model} is not existed. optional: {self.model_list}'
        for task in tasks:
            assert task in self.func_map, f'Error: Task {task} is not registered in self.func_map.'
        # input('press any key to continue.')
        print(f"Evaluation for models: {models}")
        print(f"Evaluation for tasks: {tasks}")
        for model, task in itertools.product(models, tasks):
            self.evaluate(model, task, append_mode=append_mode)
            print('Done.')

    def get_scores(self, model, task):
        task = task.replace('.json', '')
        output_file = f'{self.base_path}/outputs/{model}/{task}.json'
        output_file = glob.glob(output_file)
        if len(output_file) < 1:
            print(f'Warning: no output file found for model {model}, task {task}')
            return None
        output_file = output_file[0]
        with open(output_file, 'r') as f:
            outputs_list = json.load(f)
        scores = []
        for i, d in enumerate(outputs_list):
            if 'output' not in d: continue
            if 'score_dict' not in d: continue
            score_dict = d['score_dict']
            line = {
                "model": model,
                "ins_id": d['ins_id'],
                "param_id": d['param_id'],
                'length': d['length'],
                }
            line.update(score_dict)
            scores.append(line)
        # json.dump(scores, open(os.path.join('/data/xdwu/data/LI_tasks/Benchmarkbase1/scores/', f'{model}--{task}.json'), 'w'), indent=2)
        # df = pd.DataFrame(scores)
        # df.to_csv(os.path.join('/data/xdwu/data/LI_tasks/Benchmarkbase1/scores/', f'{model}--{task}.csv'))
        return scores

    def get_all_scores(self):
        models = self.model_list
        tasks = self.task_list
        print(f"Evaluation for models: {models}")
        print(f"Evaluation for tasks: {tasks}")
        for task in tasks:
            scores = []
            for model in models:
                print(f'Get score for {model}/{task}... ', end=" ")
                scores_ = self.get_scores(model, task)
                if scores_:
                    scores.extend(scores_)
                    print('Done.')
                else:
                    print('No results.')
            if not scores: continue
            df = pd.DataFrame(scores)
            df.to_csv(os.path.join(self.base_path, 'scores', f'{task.replace(".json", "")}.csv'), index=False)

    def get_total_score_df(self, update=False, length=0):
        if update:
            self.get_all_scores()
        tasks = list(self.func_map.keys())
        total_results_df = pd.DataFrame()
        for task in tasks:
            scores_file = glob.glob(os.path.join(self.base_path, 'scores', f'{task}.csv'))
            if not scores_file: continue
            df = pd.read_csv(scores_file[0], index_col=None)
            # if 'total_score' not in df.columns: continue
            if length:
                df = df[df['length'] == length]
            df = df.groupby('model')['total_score'].mean().reset_index()
            df.rename(columns={'total_score': task}, inplace=True)
            if not len(total_results_df):
                total_results_df = df
                continue
            total_results_df = pd.merge(total_results_df, df, on='model', how='outer')
        total_results_df.index = total_results_df['model']
        total_results_df.drop(columns=['model'], inplace=True)
        avg = total_results_df.mean(axis=1)
        weights = ScoreTools().get_total_scores_weights(total_results_df.columns)
        wavg = total_results_df.mul(weights).sum(axis=1)
        total_results_df['avg'] = avg
        total_results_df['wavg'] = wavg
        # print(total_results_df)
        return total_results_df

    def get_total_score_bylen_df(self, update=False):
        if update:
            self.get_all_scores()
        tasks = list(self.func_map.keys())
        total_results_df = pd.DataFrame()
        for task in tasks:
            scores_file = glob.glob(os.path.join(self.base_path, 'scores', f'{task}.csv'))
            if not scores_file: continue
            df = pd.read_csv(scores_file[0], index_col=None)
            if 'total_score' not in df.columns: continue
            df = df.groupby(['model', 'length'])['total_score'].mean().reset_index()
            df.rename(columns={'total_score': task}, inplace=True)
            if not len(total_results_df):
                total_results_df = df
                continue
            total_results_df = pd.merge(total_results_df, df, on=['model', 'length'], how='outer')

        # total_results_df = total_results_df.set_index(['model', 'length'])
        total_results_df['avg'] = total_results_df[[col for col in total_results_df.columns if col not in ['model', 'length']]].mean(axis=1)
        total_results_df = pd.pivot(total_results_df, index='model', columns='length', values='avg')
        total_results_df['avg'] = total_results_df.mean(axis=1)
        return total_results_df

    def get_avg_scores(self, update=False):
        if update:
            self.get_all_scores()
        tasks = list(self.func_map.keys())
        sheet_dict = {}
        for task in tasks:
            scores_file = glob.glob(os.path.join(self.base_path, 'scores', f'{task}.csv'))
            if not scores_file: continue
            # parse score point
            df = pd.read_csv(scores_file[0], index_col=None)
            score_points = list(filter(lambda x: x not in ['param_id', 'ins_id', 'length', 'model'], df.columns))
            df = df.groupby('model')[score_points].mean().reset_index()
            df.index = df['model']
            df.drop(columns=['model'], inplace=True)
            sheet_dict[task] = df

        with pd.ExcelWriter(os.path.join(self.base_path, 'results', 'score_point.xlsx')) as writer:
            dfs = []
            for task, df in sheet_dict.items():
                df.to_excel(writer, sheet_name=task, index=True)
                df = df.T
                df['task'] = task
                dfs.append(df)
            df = pd.concat(dfs)
            df = df.set_index(['task', df.index]).sort_index(axis=0)
            df.to_excel(writer, sheet_name='total_view_score_points', index=True)
            total_df = self.get_total_score_df()
            total_df.to_excel(writer, sheet_name='main_results_subtask', index=True)
            total_df = self.get_total_score_bylen_df()
            total_df.to_excel(writer, sheet_name='main_results_len', index=True)
        # return sheet_dict

    def get_scores_by_length(self, update=False):
        if update:
            self.get_all_scores()
        tasks = list(self.func_map.keys())
        sheet_dict = {}
        for task in tasks:
            scores_file = glob.glob(os.path.join(self.base_path, 'scores', f'{task}.csv'))
            if not scores_file: continue
            df = pd.read_csv(scores_file[0], index_col=None)
            if 'total_score' not in df.columns: continue
            df = df.groupby(['model', 'length'])['total_score'].mean().reset_index()
            df = pd.pivot(df, index='model', columns='length', values='total_score')
            sheet_dict[task] = df

        # total view
        total_view = sheet_dict.copy()
        for task, df in sheet_dict.items():
            df['task'] = task
            df['model'] = df.index
        df = pd.concat(sheet_dict.values())
        total_view['totalview_by_task'] = df.set_index(['task', 'model']).sort_index(axis=0)
        total_view['totalview_by_model'] = df.set_index(['model', 'task']).sort_index(axis=0)

        with pd.ExcelWriter(os.path.join(self.base_path, 'results', 'total_score_by_len.xlsx')) as writer:
            for task, df in total_view.items():
                df.to_excel(writer, sheet_name=task, index=True)
        # return sheet_dict

    def get_scorepoint_by_length(self, update=False):
        if update:
            self.get_all_scores()
        tasks = list(self.func_map.keys())
        sheet_dict = {}
        for task in tasks:
            scores_file = glob.glob(os.path.join(self.base_path, 'scores', f'{task}.csv'))
            if not scores_file: continue
            df = pd.read_csv(scores_file[0], index_col=None)
            score_points = list(filter(lambda x: x not in ['param_id', 'ins_id', 'length', 'model'], df.columns))
            # print(task)
            # print(df['format'][df['format'].apply(lambda x: isinstance(x, str))])
            df = df.groupby(['model', 'length'])[score_points].mean().reset_index()
            df['task'] = task
            df = pd.pivot(df, index=['model', 'task'], columns='length', values=score_points)
            sheet_dict[task] = df

        # total view
        total_view = sheet_dict.copy()

        with pd.ExcelWriter(os.path.join(self.base_path, 'results', 'scorepoints_by_len.xlsx')) as writer:
            for task, df in total_view.items():
                df.to_excel(writer, sheet_name=task, index=True)



    def get_all(self, models=[],tasks=[],update=True, evaluate=True, append_mode=False):
        if evaluate:
            self.evaluate_all(models=models, tasks=tasks, append_mode=append_mode)
            update = True
        if update:
            self.get_all_scores()
        self.get_avg_scores() # bytask
        self.get_scores_by_length()
        self.get_scorepoint_by_length()

if __name__ == '__main__':
    from fire import Fire
    Fire(Evaluator)

