import json
from collections import defaultdict
import glob
from typing import Tuple
import os

import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


def kmeans(embeddings, k=5):
    # 创建KMeans模型
    kmeans = KMeans(n_clusters=k)

    # 对数据进行K-means聚类
    kmeans.fit(embeddings)

    # 获取每个数据点对应的簇标签
    labels = kmeans.labels_

    # 获取每个点到各簇中心的距离
    distances = kmeans.transform(embeddings)

    # 创建一个字典保存结果，键为簇标签，值为（点的下标，距离）列表
    cluster_dict = {}

    # 遍历每个数据点，将其下标和到簇中心的距离按簇标签分类
    for idx, (label, distance) in enumerate(zip(labels, distances)):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append((idx, distance[label]))

    # 对每个簇内的列表按照距离进行排序
    for label in cluster_dict:
        cluster_dict[label].sort(key=lambda x: x[1])

    # print(cluster_dict)
    # # 打印结果
    # print("按簇分组并按距离排序的结果:")
    # for label, points in cluster_dict.items():
    #     print(f"簇 {label}: {points}")
    return cluster_dict


def sentence2embed(sentences):
    if embed_model_name == 'test':
        sentence_embeddings = np.random.rand(30, 2)
    elif embed_model_name == 'tfidf':
        # 初始化TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # 将句子转换为TF-IDF向量
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # 转换为稀疏矩阵
        sentence_embeddings = tfidf_matrix.toarray()
    elif embed_model_name == 'bge':
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('/data/xdwu/llms/bge-m3/')
        model = AutoModel.from_pretrained('/data/xdwu/llms/bge-m3/')
        model.eval()

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def sentences_cluster(sentences, num_clusters):
    embeddings = sentence2embed(sentences)
    # print(embeddings.shape)
    kmeans_dict = kmeans(embeddings, k=num_clusters)
    cluster_results = defaultdict(list)
    for label, points in kmeans_dict.items():
        for idx, dis in points:
            cluster_results[str(label)].append({'text': sentences[idx], 'distance': dis, 'ori_flag': idx == 0,
                                                'check_flag(1 checked, 2canbeused)': 0})
    return cluster_results


def get_sentences(rewrite_file_path, models=['claude-3-opus-20240229', 'gpt-4o']) -> Tuple[str, list]:
    sub_task_name = rewrite_file_path.split('/')[-1].replace('_rewrite.json', '')
    sentences = []
    with open(rewrite_file_path, 'r') as f:
        data = json.load(f)
        sentences.append(data['ori'])
        for model in models:
            parse_result = data[model]['parse_result']
            assert isinstance(parse_result, list), 'Error data 1.'
            assert all([isinstance(sentence, str) for sentence in parse_result]), 'Error data 2.'
            sentences.extend(parse_result)
    return sub_task_name, sentences


def get_prompts(dir='', check_labels=[2, 3]):
    '''Obtain the annotated task dictionary'''
    files = glob.glob(os.path.join(dir, '*.json'))
    task_prompt_map = {}
    embd_name = os.path.abspath(dir).split('/')[-1]
    for filename in files:
        data = json.load(open(filename, 'r'))
        task = filename.split('/')[-1].replace('_cluster.json', '')
        labels = ['0', '1', '2', '3', '4']
        prompts = []
        for label in labels:
            for d in data[label]:
                if d['check_flag(1 checked, 2canbeused)'] in check_labels:
                    prompts.append(d['text'])
                    break
        task_prompt_map[task] = prompts[:]
    json.dump(task_prompt_map, open(os.path.join(dir, f'{embd_name}_prompt.json'), 'w'), indent=2, ensure_ascii=False)
    return task_prompt_map


def main(input_dir, output_dir, num_clusters):
    rewrites_files = glob.glob(os.path.join(input_dir, '*.json'))

    for p in rewrites_files:
        print(p, end=' ')
        sub_task_name, sentences = get_sentences(p)
        cluster_results = {'ori_prompt': sentences[0]}
        cluster_results.update(sentences_cluster(sentences, num_clusters))
        if not os.path.exists(os.path.join(output_dir, f'{embed_model_name}')):
            os.mkdir(os.path.join(output_dir, f'{embed_model_name}'))
        json.dump(cluster_results, open(os.path.join(output_dir, f'{embed_model_name}/{sub_task_name}_cluster.json'), 'w'), indent=2, ensure_ascii=False)
        print('done.')
    # get_prompts(dir=os.path.join(output_dir, f'{embed_model_name}'))


if __name__ == '__main__':
    global embed_model_name
    embed_model_name = 'bge'
    from fire import Fire
    Fire(main())


