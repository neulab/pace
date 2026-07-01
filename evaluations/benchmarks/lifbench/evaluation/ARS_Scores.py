from typing import List
import numpy as np


class ScoreTools():
    def __init__(self):
        self.score_map = {
            "list-single_query_id": {
                'format': 1.0,
                "correct": 2.0,
                'ori': 1.0,
                "total_score": 4.0,
                },
            "list-multi_query_id": {
                # 格式得分
                "format": 2.0,
                # 生成数量得分
                "num": 3.0,
                # 正确性得分
                "correct": 3.0,
                # 顺序得分
                "order": 2.0,
                # 总分
                "total_score": 10.0,
                },
            "list-offset_query_id": {
                'format': 1.0,
                "correct": 2.0,
                'ori': 1.0,
                "total_score": 4.0,
                },
            "list-offset_query_element": {
                'format': 1.0,
                "correct": 2.0,
                'ori': 1.0,
                "total_score": 4.0,
                },
            "list-blur_offset_query_id": {
                # 是否是原文的元素出现在里面
                'ori': 1.0,
                # 顺序得分
                'position': 3.0,
                # 输出数量得分
                # 输出格式分
                'format': 1.0,
                # 总分
                "total_score": 5.0,
                },
            "list-blur_offset_query_element": {
                # 是否是原文的元素出现在里面
                'ori': 1.0,
                # 顺序得分
                'position': 3.0,
                # 输出数量得分
                # 输出格式分
                'format': 1.0,
                # 总分
                "total_score": 5.0,
                },
            "multidoc-find_dup_text": {
                # 正确性得分
                'correct': 4.0,
                # 大类数量
                'num_text': 5.0,
                # 原文尊重得分
                'ori': 6.0,
                # 格式得分
                'format': 5.0,
                # 总分
                "total_score": 20.0,
                },
            "multidoc-batch_label": {
                # candidate正确性得分
                'candi_ori': 3.0,
                # rule正确性得分
                'logit_correct': 3.0,
                'num_doc': 3.0,
                # 格式得分
                'format': 5.0,
                # 总分
                "total_score": 14.0,
                },
            "onedoc-qa": {
                "correct": 3.0,
                'format': 2.0,
                "total_score": 5.0,
                },
            "onedoc-repeat": {
                'format': 3.0,
                'ori': 2.0,
                'hit': 2.0,
                'num': 4.0,
                "correct": 3.0,
                'total_score': 14.0,
                },
            "onedoc-extract": {
                'format': 4.0,
                "correct": 4.0,
                "order": 4.0,
                "ori": 2.0,
                "total_score": 14.0,
                },
            }

    def get_total_scores_weights(self, columns: List[str]) -> List[float]:
        '''output weights according to columns
        weight:  * task_total_score / total'''
        weights = []
        for col in columns:
            if col not in self.score_map: continue
            weights.append(self.score_map[col]['total_score'])
        # assert len(weights) == len(self.score_map)
        weights = np.array(weights) / np.sum(weights)
        return weights

    def get_score(self, task, scorepoint) -> float:
        '''output weights according to columns
        weight:  * task_total_score / total'''
        return self.score_map[task][scorepoint]


def test_get_total_scores_weights():
    columns = ScoreTools().score_map.keys()
    print(columns)
    print(ScoreTools().get_total_scores_weights(columns))



if __name__ == "__main__":
    test_get_total_scores_weights()
