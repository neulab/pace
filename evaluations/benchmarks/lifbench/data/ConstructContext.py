import json
import os
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import random
from datetime import datetime, timedelta
import uuid
import base64

import nltk
import numpy as np
import tqdm
import tiktoken
from datasets import load_dataset

random.seed(0)
tokenizer = tiktoken.encoding_for_model('gpt-4o')
BASE_DIR = ''
TARGET_INPUT_LEN = [3, 6, 13, 28]


def concat_text_to_token_limit(texts: List[str], token_limit) -> str:
    tokenizer = tiktoken.encoding_for_model('gpt-4o')
    text = ''
    while len(tokenizer.encode(text)) < token_limit:
        text += '\n'.join(texts)
    return text


# 截取句子到指定token数
def truncate_text_to_target_token(text, target_token):
    tokenizer = tiktoken.encoding_for_model('gpt-4o')
    tokens = tokenizer.encode(text)
    assert len(tokens) > target_token
    return tokenizer.decode(tokens[:target_token])


def add_key_evidence_token(input_text: str) -> Tuple[str, Dict[str, List[str]]]:
    '''
    for abstract/QA data construction.
    :param input_text:
    :return: text with spacial token, evidence List: {token: (sentence, marked_order, sentence_order_idx, is_true_data)}
    '''
    # default settting： "<#Topic#>", "<@argument@>", "<!Transition!>", "<|Summary|>", "<*Evidence*>", "<-Concession->"
    sentences = nltk.sent_tokenize(input_text)
    assert len(sentences) > 23

    evidence_idxs = random.sample(list(range(len(sentences))), k=23)

    evidences = defaultdict(list)
    # add spacial token in order
    for i, idx in enumerate(evidence_idxs):
        sentence = sentences[idx].replace('\n', '')  # key sentence里不要有换行符
        if i < 4:
            evidences['#Topic#'].append((sentence, i, idx, True))
            sentence = f'<#Topic#-{i}>' + sentence + '<#Topic#>'
        elif i < 7:
            evidences['@argument@'].append((sentence, i, idx, True))
            sentence = f'<@argument@-{i - 4}>' + sentence + '<@argument@>'
        elif i == 7:
            evidences['@argument@'].append((sentence, i, idx, False))
            sentence = f'<@argument@-{i - 4}>' + sentence + '<#Topic#>'
        elif i < 13:
            evidences['!Transition!'].append((sentence, i, idx, True))
            sentence = f'<!Transition!-{i - 8}>' + sentence + '<!Transition!>'
        elif i == 13:
            evidences['!Transition!'].append((sentence, i, idx, False))
            sentence = f'<!Transition!-{i - 8}>' + sentence + '<fake_tag>'
        elif i < 18:
            evidences['*Evidence*'].append((sentence, i, idx, True))
            sentence = f'<*Evidence*-{i - 14}>' + sentence + '<*Evidence*>'
        elif i < 22:
            evidences['-Concession-'].append((sentence, i, idx, True))
            sentence = f'<-Concession--{i - 18}>' + sentence + '<-Concession->'
        elif i == 22:
            evidences['-Concession-'].append((sentence, i, idx, False))
            sentence = f'<-Concession--{i - 18}>' + sentence + '<>'
        else:
            raise f'Error i-{i} idx-{idx}'
        # print(sentence)
        sentences[idx] = sentence
    return ' '.join(sentences), evidences


class OneDocInput():
    def __init__(self):
        self.target_input_lengths = TARGET_INPUT_LEN
        self.base_dir = os.path.join(BASE_DIR, 'data/meta')
        dataset = load_dataset(os.path.join(BASE_DIR, 'corpus/paul_graham_essays'))['train']  # ['id', 'title', 'date', 'text']
        texts = [data['text'] for data in dataset]
        self.text = concat_text_to_token_limit(texts, max(self.target_input_lengths) * 1024)

    def construct_input(self) -> List[Dict[str, Union[str, dict]]]:
        inputs = {'info': [], 'inputs': []}
        for target_input_length in self.target_input_lengths:
            input_text = truncate_text_to_target_token(self.text, target_input_length * 1024)
            key_evidence, evidence_dict = add_key_evidence_token(input_text)
            inputs['inputs'].append({
                'input_text': key_evidence,
                'label': evidence_dict,
                'length': target_input_length,
                })
        json.dump(inputs, open(os.path.join(self.base_dir, 'onedoc_input.json'), 'w'), indent=2, ensure_ascii=False)
        return inputs


def generate_retrieved_list(seed=0):
    def generate_uuid_list(target_len=90000):  # 30000uuid ~ 500k tokens
        input_list = []
        input_list_text = ''
        for i in range(1, target_len + 1):
            uid = uuid.uuid1()
            input_list.append(uid.hex)
            input_list_text += f"{uid.hex}\n"
        # print('inputlist_len(token): ', len(tokenizer.encode(input_list_text)))
        # print(len(input_list))
        return input_list

    def generate_alpaca_list():
        alpaca = load_dataset(os.path.join(BASE_DIR, 'corpus/alpaca-cleaned/'))
        # alpaca = sorted(alpaca['train'].to_list(), key=lambda x: len(x['instruction']))
        alpaca['train']['instruction']
        alpaca_list = []
        for d in alpaca['train']:
            length = len(d['instruction'].split())
            if length <= 40 and length >= 5:  # 620k tokens
                alpaca_list.append(d['instruction'])
        # print(len(alpaca_list))
        # print(len(tokenizer.encode(' '.join(alpaca_list))))
        return alpaca_list

    uuid_list = generate_uuid_list()
    alpaca_list = generate_alpaca_list()
    len_uuid_list, len_alpaca_list = len(uuid_list), len(alpaca_list)
    print(len_uuid_list, len_alpaca_list, 'u_ratio:', len_uuid_list / (len_uuid_list + len_alpaca_list))
    # sample in different ratios(step 0.2)
    random.seed(seed)
    result_list = uuid_list + alpaca_list
    random.shuffle(result_list)
    return result_list


def generate_retrived_list_text_to_limit_tokens(retrieved_list, limit_tokens):
    def list2text(list_, format='ordered'):
        if format == 'ordered':
            result = ''
            for i, element in enumerate(list_):
                result += f'{i + 1}. {element}\n'
            return result
        elif format == 'unordered':
            result = ''
            for element in list_:
                result += f'- {element}\n'
            return result
        else:
            raise NotImplementedError

    tokenizer = tiktoken.encoding_for_model('gpt-4o')

    # prefill, avg 20 token/element
    k = limit_tokens // 20
    print(k)
    while len(tokenizer.encode(list2text(retrieved_list[:k]))) > limit_tokens:
        k -= 10
    while len(tokenizer.encode(list2text(retrieved_list[:k]))) < limit_tokens:
        k += 10
    k -= 10
    print(f"limited: {limit_tokens}, gen: {len(tokenizer.encode(list2text(retrieved_list[:k])))}")
    return list2text(retrieved_list[:k]), k


class ListInput():
    def __init__(self):
        self.target_input_lengths = TARGET_INPUT_LEN
        self.base_dir = os.path.join(BASE_DIR, 'data/meta')
        self.retrieved_list = generate_retrieved_list()

    def construct_input(self) -> List[Dict[str, Union[str, dict]]]:
        inputs = {'info': self.retrieved_list, 'inputs': []}
        for target_input_length in self.target_input_lengths:
            input_text, k = generate_retrived_list_text_to_limit_tokens(self.retrieved_list, target_input_length * 1024)
            inputs['inputs'].append({
                'input_text': input_text,
                'length': target_input_length,
                'label': k,
                })
        json.dump(inputs, open(os.path.join(self.base_dir, 'list_input.json'), 'w'), indent=2, ensure_ascii=False)
        return inputs


def generate_random_date(start_date="2000-01-01", end_date="2024-12-31"):
    # 将字符串日期转换为 datetime 对象
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # 计算日期之间的天数差
    days_between = (end - start).days

    # 生成一个随机天数
    random_days = random.randint(0, days_between)

    # 计算随机日期
    random_date = start + timedelta(days=random_days)

    # 格式化日期为 'yyyy-mm-dd'
    return random_date.strftime('%Y-%m-%d')


def generate_unique_id():
    # 使用 UUID4 生成唯一的 ID
    return str(uuid.uuid4())


def generate_base64_uuid():
    # 生成 UUID
    unique_id = uuid.uuid4()

    # 将 UUID 转换为 bytes 然后编码为 Base64
    base64_id = base64.urlsafe_b64encode(unique_id.bytes).rstrip(b'=').decode('utf-8')

    # 返回短的 Base64 编码字符串（长度为22个字符）
    return base64_id


def truncate(text, length):
    return tokenizer.decode(tokenizer.encode(text)[:length])


def random_sources():
    sources = ['meeting', 'news', 'govreport', 'paper', 'meeting', 'news', 'govreport', 'paper', '']
    return random.choice(sources)


def load_govreport(docnum=-1):
    p = os.path.join(BASE_DIR, 'corpus/govreport-summarization/')
    data = load_dataset(p)  # ['id', 'title', 'date', 'text']
    datas = data['train'].to_list() + data['validation'].to_list() + data['test'].to_list()
    results = []
    for d in tqdm.tqdm(datas, leave=False):
        line = {
            'iD2': generate_unique_id(),
            "id": generate_base64_uuid(),
            "text": d["report"],
            "date": generate_random_date(),
            }
        source = random_sources()
        if source:
            line['source'] = source
        text_len = len(tokenizer.encode(line['text']))
        if text_len > 300 and text_len <= 500:
            results.append(line)
        else:
            line['text'] = truncate(line['text'], 200)
            results.append(line)
        if docnum > 0 and len(results) >= docnum:
            break
    return results


def load_qmsum(docnum=-1):  # 太长了得截取一下 2500 17 34786
    p = os.path.join(BASE_DIR, 'corpus/qmsum-cleaned/')
    data = load_dataset(p)  # ['id', 'title', 'date', 'text']
    # print(data)
    datas = data['train'].to_list() + data['validation'].to_list() + data['test'].to_list()
    results = []
    for d in tqdm.tqdm(datas, leave=False):
        line = {
            'iD2': generate_unique_id(),
            "id": generate_base64_uuid(),
            "title": d['id'],
            "text": d["input"],
            "date": generate_random_date(),
            }
        source = random_sources()
        if source:
            line['source'] = source
        text_len = len(tokenizer.encode(line['text']))
        if text_len > 300 and text_len <= 500:
            results.append(line)
        else:
            line['text'] = truncate(line['text'], 200)
            results.append(line)
        if docnum > 0 and len(results) >= docnum:
            break
    return results


def load_paul(docnum=-1):  # 57 43656
    p = os.path.join(BASE_DIR, 'corpus/paul_graham_essays/')
    data = load_dataset(p)['train']  # ['id', 'title', 'date', 'text']
    results = []
    for d in data:
        line = {
            "id": generate_base64_uuid(),
            'iD2': generate_unique_id(),
            "title": d['title'],
            "text": d["text"],
            "date": d["date"],
            }
        source = random_sources()
        if source:
            line['source'] = source
        text_len = len(tokenizer.encode(line['text']))
        if text_len > 300 and text_len <= 500:
            results.append(line)
        else:
            line['text'] = truncate(line['text'], 200)
            results.append(line)
        if docnum > 0 and len(results) >= docnum:
            break
    return results


def load_xsum(docnum=-1):  # much
    p = os.path.join(BASE_DIR, 'corpus/xsum/')
    data = load_dataset(p)  # ['id', 'title', 'date', 'text']
    datas = data['train'].to_list() + data['validation'].to_list() + data['test'].to_list()
    results = []
    for d in tqdm.tqdm(datas, leave=False):
        line = {
            'iD2': generate_unique_id(),
            "id": generate_base64_uuid(),
            "title": d['id'],
            "text": str(d["dialogue"]),
            "date": generate_random_date(),
            }
        source = random_sources()
        if source:
            line['source'] = source
        text_len = len(tokenizer.encode(line['text']))
        if text_len > 300 and text_len <= 500:
            results.append(line)
        else:
            line['text'] = truncate(line['text'], 200)
            results.append(line)
        if docnum > 0 and len(results) >= docnum:
            break
    return results


def doc_sample():
    doc_list = []
    data = load_qmsum(400)
    doc_list.extend(data)
    data = load_paul(400)
    doc_list.extend(data)
    data = load_xsum(400)
    doc_list.extend(data)
    data = load_govreport(400)
    doc_list.extend(data)
    random.shuffle(doc_list)
    return doc_list


def doc2text(docid, doc_dict):
    text = f'******************** doc-{docid} ********************\n'
    keys = ['id', 'iD2', 'title', 'text', 'date', 'source']
    random.shuffle(keys)
    for k in keys:
        v = doc_dict.get(k, None)
        if not v: continue
        text += f'{k}: {v}\n'
    return text


def docs2text(doc_list):
    text = ''
    for doc_id, doc_dict in enumerate(doc_list, start=1):
        text += doc2text(doc_id, doc_dict)
        text += '\n\n'
    return text


def make_duplcated_list(doc_list, dupli_ratio=0.25, doc_num=5):
    def replace_doc(i, j):
        '''replace j in doclist by i, keep id'''
        # id_ = doc_list[j]['id']
        doc_list[j]['text'] = doc_list[i]['text']
        # doc_list[j]['id'] = id_

    duplicated_num_per_doc = (int(len(doc_list) * dupli_ratio) // doc_num) + 1
    dupli_idxs = [0, 2] + random.sample(list(range(3, len(doc_list))), k=duplicated_num_per_doc * doc_num - 2)

    labels = np.array(dupli_idxs, dtype=int).reshape(doc_num, -1).tolist()
    labels = [list(sorted(l)) for l in labels]
    print(labels)
    for labels_ in labels:
        ori_idx = labels_[0]
        for idx in labels_[1:]:
            replace_doc(ori_idx, idx)
    return doc_list, labels


def generate_doc_list_text_to_limit_tokens(doc_list, limit_tokens):
    tokenizer = tiktoken.encoding_for_model('gpt-4o')

    # prefill, avg 20 token/element
    k = limit_tokens // 700
    print(k)
    while len(tokenizer.encode(docs2text(doc_list[:k]))) > limit_tokens:
        k -= 1
    while len(tokenizer.encode(docs2text(doc_list[:k]))) < limit_tokens:
        k += 1
    k -= 1
    print(f"limited: {limit_tokens}, gen: {len(tokenizer.encode(docs2text(doc_list[:k])))}, doc_num:{k}")
    return docs2text(doc_list[:k]), k


class MultiDocInput():
    def __init__(self):
        self.target_input_lengths = TARGET_INPUT_LEN
        self.base_dir = os.path.join(BASE_DIR, 'data/meta')
        doc_list = doc_sample()
        self.doc_list, self.labels = make_duplcated_list(doc_list, dupli_ratio=0.25, doc_num=5)

    def construct_input(self) -> List[Dict[str, Union[str, dict]]]:
        inputs = {'info': {'doc_list': self.doc_list, 'dupli_idxs': self.labels}, 'inputs': []}
        for target_input_length in self.target_input_lengths:
            input_text, k = generate_doc_list_text_to_limit_tokens(self.doc_list, target_input_length * 1024)
            inputs['inputs'].append({
                'input_text': input_text,
                'length': target_input_length,
                'label': k,
                })
        json.dump(inputs, open(os.path.join(self.base_dir, 'multidoc_input.json'), 'w'), indent=2, ensure_ascii=False)
        return inputs


def main(base_dir: str,
         context_lengths: List, ):
    BASE_DIR = base_dir
    TARGET_INPUT_LEN = context_lengths

    OneDocInput().construct_input()
    ListInput().construct_input()
    MultiDocInput().construct_input()


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
