# multidoc
# ok
import re
import json


def judge_multidoc_batch_label(output_dict):
    score_dict = {
        # candidate正确性得分
        'candi_ori': 0.0,  # 3.0

        # rule正确性得分
        'logit_correct': 0.0,  # 3.0

        'num_doc': 0.0,  # 3.0

        # 格式得分
        'format': 0.0,  # 5.0

        # 总分
        "total_score": 0.0  # 14.0
        }
    expected_element = output_dict["param"]["candidates"]

    cadidate = output_dict["param"]["candidates"]

    rule_key = output_dict["param"]["rule_key"]

    # 全部的文件
    all_doc = output_dict["param"]["doc_keys"]

    # 创建一个字典来映射每个字符到其在 rule_key 中的索引
    order_map = {char: index for index, char in enumerate(rule_key)}

    sorted_candidates = sorted(cadidate, key=lambda word: order_map[word[0]])

    # 处理出正确的答案
    true_answer = []
    for _ in all_doc:
        flag = 0
        if "title" not in _:
            flag += 2
        if "source" not in _:
            flag += 1
        true_answer.append(sorted_candidates[flag])

    # LLM输出
    llm_output = output_dict["output"].strip()

    # 获取有多少个item
    num_docs = output_dict['label']

    # 正则表达式模式
    pattern = r'\{[^\]]+\}'

    # 查找所有匹配项
    matches = re.findall(pattern, llm_output)

    # 如果匹配到了超过2个：
    parse_output = ""
    llm_output_ = llm_output
    for i in range(2):
        try:
            parse_output = json.loads(llm_output_)
            score_dict['format'] += 2.0 if i == 0 else 1.0
            break
        except:
            if i == 0:
                pattern = r'\{[^\]]+\}'
                # 查找所有匹配项
                matches = list(re.findall(pattern, llm_output_))  # 选最长的那个
                if not matches:
                    break
                matches.sort(key=lambda x: len(x), reverse=True)
                llm_output_ = matches[0]
    if parse_output and isinstance(parse_output, dict):
        score_dict['format'] += 1.0
    if parse_output and (isinstance(parse_output, dict) or isinstance(parse_output, list)):
        num_docs_output = len(parse_output)
    else:
        num_docs_output = llm_output.count('doc')
    score_dict['num_doc'] = max(0, (1.0 - (abs(num_docs_output - num_docs) / num_docs))) * 2.5
    score_dict['num_doc'] += 0.5 if num_docs_output == num_docs else 0.0  # 额外奖励
    # 查看有多少关键字在里面
    key_count = sum([1.0 if f'doc{k}' in llm_output else 0.0 for k in range(1, num_docs + 1)])
    score_dict["format"] += (key_count / num_docs) * 2.0

    # 进行多串的匹配
    for match in matches:
        if (len(match.split(',')) == num_docs):
            # 按照，分割
            match = match.split(',')
            sum_candidate = 0
            sum_rule = 0
            for index, str in enumerate(match):
                flag = False
                # 查看是否在候选集里
                if true_answer[index] in str:
                    sum_rule += 1
                for f in cadidate:
                    if f in str:
                        flag = True
                        break
                if flag == True:
                    sum_candidate += 1
            # 最后算得分
            score_dict["candi_ori"] = sum_candidate * 1.0 / num_docs * 3.0
            score_dict["logit_correct"] = sum_rule * 1.0 / num_docs * 3.0
            break
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 14.0
    score_dict['candi_ori'] /= 3.0
    score_dict['logit_correct'] /= 3.0
    score_dict['format'] /= 5.0
    score_dict['num_doc'] /= 3.0
    return score_dict


# 将dict或者list转化为map，反向建立索引
def op_answer2map(answer, need, flag, t, prompt):
    # 创建一个字典来存储数据，方便映射
    mp = {}
    num = 0
    for _ in range(0, len(answer)):
        run = answer[_]
        if (flag == 0):
            for d in range(0, len(run)):
                tmp = []
                for t in need:
                    tmp.append(str(run[d][t]).replace("\"", ""))
                mp[tuple(tmp)] = _
        else:
            for d in range(0, len(run)):
                run[d] = str(run[d])
                if str(run[d]) in prompt:
                    num += 1
                else:
                    # print(str(run[d]))
                    pass
            mp[tuple(run)] = t
    return mp, num


def judge_multidoc_find_dup_text(output_dict):
    score_dict = {
        # 正确性得分
        'correct': 0.0,  # 4.0

        # 大类数量
        'num_text': 0.0,  # 5.0

        # 原文尊重得分
        'ori': 0.0,  # 6.0

        # 格式得分
        'format': 0.0,  # 5.0

        # 总分
        "total_score": 0.0  # 20.0
        }
    # LLM输出
    llm_output = output_dict["output"].strip()
    llm_prompt = output_dict["prompt"].strip()
    llm_input = get_input_text(output_dict['prompt'].strip(), start='\n\n Documents:\n', end='\n\n Instructions:')
    need = output_dict["param"]["info_keys"].strip().split(' ')
    answer = output_dict["param"]["dup_infos"]
    # examples = "[[\"iD2_1\"], [\"iD2_2\"], [\"iD2_3\"]]\n[[\"iD2_4\"], [\"iD2_5\"]], [[\"id_1\"], [\"id_2\"], [\"id_3\"]]\n[[\"id_4\"], [\"id_5\"]], [[\"id_1\", \"title_1\"], [\"id_2\", \"title_2\"], [\"id_3\", None]]\n[[\"id_4\", \"title_4\"], [\"id_5\", \"title_5\"]], [[\"iD2_1\", \"date_1\"], [\"iD2_2\", \"date_2\"], [\"iD2_3\", \"date_3\"]]\n[[\"iD2_4\", \"date_4\"], [\"iD2_5\", \"date_5\"]][[\"id_1\", \"source_1\", \"title_1\"], [\"id_2\", None, \"title_2\"], [\"id_3\", \"source_3\", None]]\n[[\"id_4\", \"source_4\", \"title_4\"], [\"id_5\", \"source_5\", \"title_5\"]]"

    # 如果是空输出，直接返回
    if not llm_output:
        return score_dict

    # 格式得分，输出是否只输出[],而不掺杂其他的东西
    if '[' in llm_output and ']' in llm_output:
        score_dict["format"] += 0.5
    # 查看有无多余输出
    l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
    if l_idx == 0 and r_idx == len(llm_output) - 1:
        score_dict['format'] += 0.5
    parse_outputs = []
    parse_flag = False  # 是否存在可以被解析的内容
    all_parse_flag1 = True  # 外层list解析
    all_parse_flag2 = True  # 内层list解析
    pattern = r'\[\[.*?\]\]'
    matches = re.findall(pattern, llm_output)
    if len(matches) == len(answer):
        score_dict["num_text"] = 1.0
    score_dict["num_text"] += max(0, (1.0 - (abs(len(matches) - len(answer)) / (len(answer) + 2)))) * 4.0
    if len(matches) == len(llm_output.split('\n')):
        score_dict['format'] += 1.0
    for line in matches:
        try:
            l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
            line = line[l_idx:r_idx + 1]
            parse_line = json.loads(line)
            parse_flag = True
            if isinstance(parse_line, list) and all([isinstance(l, list) for l in parse_line]):
                parse_outputs.append(parse_line)
            else:
                all_parse_flag2 = False
        except:
            all_parse_flag1 = False
            continue
    if parse_flag:
        score_dict['format'] += 1.0
    if all_parse_flag1:
        score_dict['format'] += 1.0
    if all_parse_flag2:
        score_dict['format'] += 1.0

    # ori 1 查看解析出来的元素有多少是属于原文的
    ori_count = 0
    total = 0

    for parse_line in parse_outputs:
        for l in parse_line:
            total += len(l)
            ori_count += sum([1.0 if (str(e) in llm_input or e == None) else 0 for e in l])

    score_dict['ori'] = (ori_count / total) * 3.0 if total else 0.0
    llm_prompt = llm_prompt.replace("\"", "")
    llm_output = llm_output.replace("null", "None")
    # correct part 1 看目标有多少是在output中出现了 3.0
    contents = []
    for l in answer:
        for d in l:
            t = list(d.values())
            contents.extend(t)
    count = 0
    for content in contents:
        try:
            decoded_output = llm_output.encode('utf-8').decode('unicode_escape')
        except (UnicodeDecodeError, UnicodeEncodeError):
            decoded_output = llm_output
        if str(content) in decoded_output:
            count += 1

    score_dict['correct'] += (count * 3.0 / len(contents))
    llm_output = llm_output.replace("\n", ",")
    llm_output = llm_output.replace("\\\"", "")
    llm_output = llm_output.replace("\\", ",")
    llm_output = get_input_text(llm_output, start='\n\n Documents:\n', end='\n\n Instructions:')
    # correct part 2
    # 为了以防万一，匹配一下，把所有框框全都匹配出来
    pattern = r'\[\[.*?\]\]'
    matches = re.findall(pattern, llm_output)

    # 全匹配正确性比对
    # 处理出answer的map
    mp1, tmp1 = op_answer2map(answer, need, 0, 1, llm_prompt)
    # 处理出output的map
    mp2 = {}
    t = 0
    sum_ = true_item = 0
    for _ in matches:
        _ = _.replace(",\"", "")
        if is_valid_eval(_) == True:
            _ = eval(_)

            sum_ += len(_) * len(_[0])

            mp_tmp, tmp = op_answer2map(_, need, 1, t, llm_prompt)
            for x, y in mp_tmp.items():
                mp2[x] = y
            true_item += tmp
            t += 1
    if sum_ != 0:
        score_dict["ori"] += true_item * 3.0 / sum_
    true_pair = 0

    for x, y in mp1.items():
        for x2, y2 in mp1.items():
            # 两个其中一个不在键值对里
            if (x not in mp2):
                continue
            if (x2 not in mp2):
                continue
            if y == y2 and mp2[x] == mp2[x2]:
                true_pair += 1
            if y != y2 and mp2[x] != mp2[x2]:
                true_pair += 1

    score_dict["correct"] += true_pair * 1.0 / (len(mp1) * len(mp1))
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 20.0
    score_dict['format'] /= 5.0
    score_dict['correct'] /= 4.0
    score_dict['ori'] /= 6.0
    score_dict['num_text'] /= 5.0
    return score_dict



def get_input_text(prompt, start='\nInput text: ', end='\nOutput:'):
    r = prompt.split(start)[-1]
    if '\nOutput:' in r:
        r = r.split(end)[0]
    assert len(r) > 0
    return r.strip()


def parse_oredered_input_list_text(inptext, start=1):
    inptext = '\n' + inptext.strip()
    parse_list = []
    p = start
    # bar = tqdm.tqdm(total=len(inplist))
    idxs = []
    while True:
        tag = f'\n{p}. '
        # bar.update(1)
        start_idx = idxs[-1] if idxs else 0
        idx = inptext.find(tag, start_idx)
        if idx != -1:
            idxs.append(idx)
        else:
            break
        p += 1
    for i in range(len(idxs) - 1):
        parse_list.append(inptext[idxs[i]: idxs[i + 1]].split('. ', maxsplit=1)[-1])
    parse_list.append(inptext[idxs[-1]:].split('. ', maxsplit=1)[-1])
    return parse_list


def find_json_list(s):
    s_ = ' '.join(s.split('\n')).strip()
    exp = re.compile(r'(\[.*?\])')
    r = exp.findall(s)
    if r:
        try:
            # 尝试解析找到的第一个JSON字符串
            return json.loads(r[0])
        except json.JSONDecodeError as e:
            # 处理解析错误
            # print(f"JSON Decode Error: {e}")
            return None
    else:
        return None


def find_json_dict(s):
    s = ' '.join(s.split('\n')).strip()
    exp = re.compile(r'(\{.*?\})')
    r = exp.findall(s)
    return json.loads(r[0]) if r else None


def directly_load_json(output):
    try:
        # 尝试解析找到的第一个JSON字符串
        return json.loads(output)
    except json.JSONDecodeError as e:
        # 处理解析错误
        # print(f"JSON Decode Error: {e}")
        return None


def extract_in_brackets(text):
    match = re.search(r'<([^<>]+)>', text)
    if match:
        return match.group(1)
    else:
        # print(text)
        return None


# 比较两个列表中有多少是重合的
def compare_list(list_target, list_b):
    # 无法解析出正确结果时
    if list_b is None or not isinstance(list_b, list):
        return 0.0
    # 除零0的错误要小心
    if len(list_target) == 0:
        if len(list_b) == 0:
            return 1.0
        else:
            return 0.0
    length = min(len(list_target), len(list_b))

    right = 0
    for index in range(0, length):
        # print(list_target[index][0])
        count = str(list_b[index]).count(list_target[index][0])
        if count:
            if list_target[index][3]:
                right += 1.0 / count
            # 如果不是key sentence加上只加一半的分
            else:
                right += 0.5 / count
    return right * 1.0 / len(list_target)


def contain_list(target_list, output):
    if len(target_list) == 0:
        if output == "[]":
            return 1.0
        else:
            return 0.0
    right = 0
    length = len(target_list)
    for index in range(0, len(target_list)):
        count = output.count(target_list[index][0])
        if count != 0:
            # 如果是key sentence
            if target_list[index][3]:
                right += 1.0 / count
            # 如果不是key sentence
            else:
                right += 0.5 / count
    return right * 1.0 / len(target_list)


# one_doc

def judge_onedoc_extract(output_dict):
    score_dict = {
        'format': 0.0,  # 4.0 格式分数
        "correct": 0.0,  # 4.0 包含正确分数, 全面性
        "order": 0.0,  # 4.0 输出包含正确的顺序
        "ori": 0.0,  # 2.0
        "total_score": 0.0,  # 14.0
        }
    llm_output = output_dict['output'].strip()
    label = output_dict["label"]
    target_att = output_dict["param"]["type"][1:-1]
    # 如果是空输出，直接返回
    if llm_output == '':
        return score_dict

    # 能否匹配json格式
    num_output_sentences = 0
    if is_valid_json(llm_output):
        # 字典里都是str格式
        t = json.loads(llm_output)
        if isinstance(t, list):
            if all([isinstance(i, str) for i in t]) or (target_att == '|Summary|' and t == []):
                score_dict['format'] = 4.0
            else:
                score_dict['format'] = 3.0
            num_output_sentences = len(t)
        else:
            score_dict['format'] = 1.0
    else:  # 找出最外围的两个括号再处理一遍
        l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
        llm_output_ = llm_output[l_idx:r_idx + 1]
        if is_valid_json(llm_output_):
            # 字典里都是str格式
            t = json.loads(llm_output_)
            if isinstance(t, list):
                if all([isinstance(i, str) for i in t]) or (target_att == '|Summary|' and t == []):
                    score_dict['format'] = 3.0
                else:
                    score_dict['format'] = 2.0
                num_output_sentences = len(t)
            else:
                score_dict['format'] = 1.0

    # 特殊处理 *
    if target_att == '|Summary|' and llm_output == '[]':
        score_dict = {
            'format': 1.0,  # 4.0 格式分数
            "correct": 1.0,  # 4.0 包含正确分数, 全面性
            "order": 1.0,  # 3.0 输出包含正确的顺序
            "ori": 1.0,  # 2.0
            "total_score": 1.0,  # 13.0
            }
        return score_dict

    # ori 辛苦分，以及顺序分
    try:
        llm_output = llm_output.encode('utf-8').decode('unicode_escape')
    except:
        llm_output = llm_output[:-1].encode('utf-8').decode('unicode_escape')


    key_sentences = []  # 按顺序获得所有key sentences
    for att, key_infos in label.items():
        key_sentences.extend([info[0] for info in key_infos])

    t_pos = 0
    correct_pos_count = 0
    total_count = 0
    for sentence in key_sentences:
        p = llm_output.find(sentence)
        if p == -1: continue
        total_count += 1
        correct_pos_count += (p >= t_pos)
        t_pos = min(p, t_pos)  # 尽可能多给分
    score_dict['order'] = correct_pos_count / total_count * 4.0 if total_count else 0.0
    score_dict['ori'] = 2.0 if total_count >= 1 else 0.0


    true_sentences = []
    fake_sentences = []
    for e, _, _, istrue in label.get(target_att, []):
        if istrue:
            true_sentences.append(e)
        else:
            fake_sentences.append(e)
    hit_count = 0
    for sentence in true_sentences:
        p = llm_output.find(sentence)
        if p == -1: continue
        hit_count += 1
    if max(total_count, num_output_sentences, len(true_sentences)):
        score_dict['correct'] = hit_count / max(total_count, num_output_sentences, len(true_sentences)) * 3.5  # 生成太多也会被惩罚
    else:
        score_dict['correct'] = 0.0
    assert len(fake_sentences) in [0, 1]
    if fake_sentences == [] or fake_sentences[0] not in llm_output:
        score_dict['correct'] += 0.5

    # 求平均分数
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 14.0

    score_dict['format'] /= 4.0
    score_dict['correct'] /= 4.0
    score_dict['order'] /= 4.0
    score_dict['ori'] /= 2.0
    return score_dict


def count_outer_brackets(s, content_list):
    # 将 content_list 转换成正则表达式中的一个部分
    # 使用 | （或）运算符将列表中的每个字符串连接起来
    content_pattern = '|'.join(re.escape(item) for item in content_list)

    # 创建一个正则表达式，用于匹配包含列表中任意一个字符串的最外层尖括号对
    pattern = fr'<[^<]({content_pattern})[^>]*>'

    # 查找所有匹配的部分
    matches = re.findall(pattern, s)
    return len(matches)


def compare_sentences(target_sentence, sentence_b):
    if sentence_b.find(target_sentence) != -1:
        return True


def judge_onedoc_repeat(output_dict):
    score_dict = {
        'format': 0.0,  # 3.0 判断 sep 符号和“\n”
        'ori': 0.0,  # 2.0 有多少来自原文的内容
        'hit': 0.0,  # 2.0 命中了 key sentence的数量
        'num': 0.0,  # 4.0 数量是否正确
        "correct": 0.0,  # 3.0 类型是否正确
        'total_score': 0.0  # 14.0
        }
    llm_output = output_dict['output'].strip()
    label = output_dict["label"]
    sep = output_dict["param"]["sep"]
    expected_num = output_dict["param"]["num"]
    att_list = ["#Topic#", "@argument@", "!Transition!", "|Summary|", "*Evidence*", "-Concession-"]
    if not llm_output:
        return score_dict

    # 计算ori数量
    hit_count = 0
    fake_count = 0
    # 避免 text 内 '\n' 影响，对label 和 output先做一个预处理.(替换成无回车版本)
    for att, key_infos in label.items():
        for i, info in enumerate(key_infos):
            text = info[0]
            iskey = info[-1]
            if text not in llm_output: continue
            hit_count += 1
            fake_count += 1 if not iskey else 0
            if '\n' not in text: continue
            pure_text = text.replace('\n', '')
            llm_output = llm_output.replace(text, pure_text)
            label[att][i][0] = pure_text
    if hit_count >= 0:  # 有原文文本
        score_dict['ori'] += 1.0
    if any(att in llm_output for att in att_list):
        score_dict['ori'] += 1.0
    score_dict['hit'] += min(1.0, (hit_count - 0.5 * fake_count) / expected_num) * 2.0  # fake的内容要惩罚

    # format part 1 1.0
    if expected_num == 1 or '\n' in llm_output:
        score_dict['format'] += 0.5
    if sep in llm_output:
        score_dict['format'] += 0.5

    # 判断num数量，从sep数，list解析数分析数量是否匹配 4.0
    parse_list = list(filter(lambda x: x, llm_output.split('\n')))
    num_sep = llm_output.count(sep)
    if num_sep == expected_num:
        score_dict['num'] += 1.0
    if len(parse_list) == expected_num:
        score_dict['num'] += 1.0
    if hit_count == expected_num:
        score_dict['num'] += 1.0
    score_dict['num'] += sum([num_sep == len(parse_list), num_sep == hit_count, hit_count == len(parse_list)]) / 3

    # format part2 2.0
    true_count = 0
    total = 0  # 有sep的line的数量
    for line in parse_list:
        if sep not in llm_output: continue
        total += 1
        if len(line.split(sep)) == 2:
            true_count += 1
    score_dict['format'] += min(1.0, true_count / max(total, expected_num)) * 2.0

    # correct 判断命中句子的类型是否正确
    correct_count = 0
    total = 0  # 有sep的line的数量
    for line in parse_list:
        if sep in line:
            sentence, att_str = line.split(sep, maxsplit=1)
        else:
            sentence = line.strip()
            att_str = line.strip()
        possible_atts = filter(lambda x: x in att_str, att_list)
        total += 1 if possible_atts else 0
        possible_sentences = []
        for att in possible_atts:  # 从可能的属性获得可能的sentences
            possible_sentences.extend([key_infos[0] for key_infos in label.get(att, [])])
        if any([min_edi_dis(sentence, psentence) <= len(psentence) // 6 + 3 for psentence in possible_sentences]):
            correct_count += 1
    score_dict['correct'] += (correct_count / total) * 3.0 if total else 0

    # 求平均分数
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 14.0
    score_dict['format'] /= 3.0
    score_dict['ori'] /= 2.0
    score_dict['hit'] /= 2.0
    score_dict['num'] /= 4.0
    score_dict['correct'] /= 3.0
    return score_dict


def judge_onedoc_qa(output_dict):
    score_dict = {
        "correct": 0.0,  # 3.0
        'format': 0.0,  # 2.0
        "total_score": 0.0  # 5.0
        }
    llm_output = output_dict['output'].strip()
    options = output_dict["param"]['options']
    evidence = output_dict["param"]['evidence'].strip()
    label = output_dict["label"]

    # 拿到真实标签
    iskey = False # 是否是标签
    istruekey = False  # 是否是真实标签
    for att, key_infos in label.items():
        for text, _, _, truekey in key_infos:
            if text.strip() == evidence:
                iskey = True
                istruekey = truekey
                break
        if iskey: break

    # format 2.0 长度， 范式
    if len(llm_output) == len(options[0]) or len(llm_output) == len(options[1]):
        score_dict['format'] += 1.0
    elif len(llm_output) > max(len(options[0]), len(options[1])) + 3:
        score_dict['format'] += 0.5

    if options[0] in llm_output and options[1] in llm_output:
        score_dict['format'] += 0.5
    elif options[0] in llm_output or options[1] in llm_output:
        score_dict['format'] += 1.0

    # correct
    if options[0] in llm_output and options[1] in llm_output:
        score_dict['correct'] = 0.5
    elif not (options[0] in llm_output or options[1] in llm_output):
        score_dict['correct'] = 0.0
    else:
        llm_answer = True if options[0] in llm_output else False
        if istruekey:
            score_dict['correct'] = 0.1 if llm_answer ^ iskey else 3.0
        else:
            score_dict['correct'] = 2.0 if llm_answer else 3.0

    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 5.0
    score_dict['format'] /= 2.0
    score_dict['correct'] /= 3.0
    return score_dict


# list input
def get_input_text(prompt, start='\nInput text: ', end='\nOutput:'):
    r = prompt.split(start)[-1]
    if end in r:
        r = r.split(end)[0]
    assert len(r) > 0
    return r.strip()



def find_json_dict(s):
    s = ' '.join(s.split('\n')).strip()
    exp = re.compile(r'(\{.*?\})')
    r = exp.findall(s)
    return json.loads(r[0]) if r else None


# json格式是否可以解析成功
def is_valid_json(input_string):
    try:
        json_object = json.loads(input_string)
        return True
    except ValueError as e:
        return False


# 是否能eval成一个list
def is_valid_eval(expression):
    try:
        result = eval(expression)
        return True
    except Exception as e:
        # 捕获所有其他可能的异常
        return False


def min_edi_dis(n, m):
    len_s1, len_s2 = len(n), len(m)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if (n[i - 1] == m[j - 1]):
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[i][j]


# ok
def judge_label_equal_output(output_dict, assistant=False):
    score_dict = {
        'format': 0.0,  # 1.0
        "correct": 0.0,  # 2.0
        'ori': 0.0,  # 1.0
        "total_score": 0.0  # 4.0
        }

    # LLM输出
    llm_output = output_dict["output"].strip()
    bias = output_dict["param"].get("bias", 0)  # 有bias 就是offset的任务
    if bias == 0:
        expected_element = output_dict["param"]["element"]
    elif bias == 1:
        expected_element = output_dict["param"]["post_element"]
    elif bias == -1:
        expected_element = output_dict["param"]["pre_element"]

    input_list_str = get_input_text(output_dict["prompt"], start='List to be retrieved:\n', end='\nInstruction:')
    input_list = parse_oredered_input_list_text(input_list_str)
    input_list_lengths = set(len(x) for x in input_list)
    # 如果是空元素，直接返回
    if llm_output == '':
        return score_dict

    # 如果直接是一个元素，可用，就给格式分
    if min(input_list_lengths) <= len(llm_output) <= max(input_list_lengths):
        score_dict['format'] = 0.2

    # 如果存在至少一个元素在其中，那么就是原文尊重的
    count = 0

    for idx, element in enumerate(input_list):
        count += llm_output.count(element)

    if count == 0:
        return score_dict
    elif count == 1:
        score_dict["ori"] = 1.0
        score_dict["format"] += 0.8 if llm_output in input_list else 0.3
    else:  # >=2
        score_dict["ori"] = 1.0

    # 正确性得分
    if expected_element in llm_output:
        score_dict["correct"] = 2.0
    # 单纯输出错误，可能会错几个字母导致答案问题。
    elif (min_edi_dis(expected_element, llm_output) <= len(expected_element) / 6 + 3):
        score_dict["correct"] = 1.0
    else:
        score_dict["correct"] = 0

    # 计算总分
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 4.0
    score_dict['correct'] /= 2.0
    return score_dict


# 匹配所有括号
# json格式预处理
# 顺序
def judge_labels_equal_outputs(output_dict):
    score_dict = {
        # 格式得分
        "format": 0.0,  # 2.0
        # 生成数量得分
        "num": 0.0,  # 3.0
        # 正确性得分
        "correct": 0.0,  # 3.0
        # 顺序得分
        "order": 0.0,  # 2.0
        # 总分
        "total_score": 0.0,  # 10.0
        }
    # # 答案
    expected_elements = output_dict["param"]["elements"]
    ## LLM输出
    llm_output = output_dict["output"].strip()
    input_list_str = get_input_text(output_dict["prompt"], start='List to be retrieved:\n', end='\nInstruction:')
    input_list = parse_oredered_input_list_text(input_list_str)
    num_output_elements = -1

    # 如果是空输出，直接返回
    if llm_output == '':
        return score_dict

    # 能否匹配json格式
    if is_valid_json(llm_output):
        # 字典里都是str格式
        t = json.loads(llm_output)
        if isinstance(t, list):
            if all([isinstance(i, str) for i in t]):
                score_dict['format'] = 2.0
            else:
                score_dict['format'] = 1.5
            num_output_elements = len(t)
        else:
            score_dict['format'] = 0.5
    else:  # 找出最外围的两个括号再处理一遍
        l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
        llm_output_ = llm_output[l_idx:r_idx + 1]
        if is_valid_json(llm_output_):
            # 字典里都是str格式
            t = json.loads(llm_output_)
            if isinstance(t, list):
                if all([isinstance(i, str) for i in t]):
                    score_dict['format'] = 1.5
                else:
                    score_dict['format'] = 1.0
                num_output_elements = len(t)
            else:
                score_dict['format'] = 0.5

    # 判断生成的元素数量:
    if num_output_elements == -1:
        num_output_elements = 0
        for element in input_list:
            num_output_elements += llm_output.count(element)
    if num_output_elements == len(expected_elements):
        score_dict['num'] += 1
    score_dict['num'] += max(0, (1.0 - (abs(num_output_elements - len(expected_elements)) / len(expected_elements)))) * 2

    # 判断正确数量
    llm_output = llm_output.encode('utf-8').decode('unicode_escape')

    match_elements = []
    for element in expected_elements:
        if element in llm_output:
            match_elements.append(element)

    score_dict['correct'] = len(match_elements) / len(expected_elements) * 3.0

    # 判断是否有序
    if len(match_elements) <= 1:
        score_dict['order'] = 1.0
    else:
        order_flag = True
        p = 0
        for element in match_elements:
            t = llm_output.find(element)
            if t >= p:
                p = t
            else:
                order_flag = False
                break
        score_dict['order'] = 2.0 if order_flag else 0.0

    # 计算总分
    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 10.0
    score_dict['format'] /= 2.0
    score_dict['num'] /= 3.0
    score_dict['correct'] /= 3.0
    score_dict['order'] /= 2.0
    return score_dict


# ok
def judge_list_input_blur_offset_query(output_dict):
    score_dict = {
        # 是否是原文的元素出现在里面
        'ori': 0.0,  # 1.0
        # 顺序得分
        'position': 0.0,  # 3.0
        # 输出数量得分
        # 输出格式分
        'format': 0.0,  # 1.0
        # 总分
        "total_score": 0.0  # 5.0
        }
    prompt = output_dict["prompt"]
    output = output_dict["output"]
    label_element_idx = output_dict["param"]['id'] - 1
    offset = output_dict["param"]["bias"]
    llm_output = output.strip()
    input_list_str = get_input_text(prompt, start='List to be retrieved:\n', end='\nInstruction:')
    input_list = parse_oredered_input_list_text(input_list_str)
    input_list_lengths = set(len(x) for x in input_list)
    # 如果是空元素，直接返回
    if llm_output == '':
        return score_dict

    # 如果直接是一个元素，可用，就给格式分
    if min(input_list_lengths) <= len(llm_output) <= max(input_list_lengths):
        score_dict['format'] = 0.2

    # 有哪些list的元素在输出里面
    in_idx = []
    for idx, element in enumerate(input_list):
        if element in llm_output:
            in_idx.append(idx)
    if len(in_idx) == 0:  # 没有输出相关元素:
        return score_dict
    elif len(in_idx) == 1:  # 输出一个，数量刚好
        score_dict['format'] += 0.8 if llm_output == input_list[in_idx[0]] else 0.3
        score_dict['ori'] = 1.0
    else:  # 超过两个
        score_dict['ori'] = 1.0

    # 对比idx 看位置对不对
    if offset == 1:
        if all([i > label_element_idx for i in in_idx]):
            score_dict['position'] = 3.0
        elif all([i >= label_element_idx for i in in_idx]):
            score_dict['position'] = 1.0
        else:
            score_dict['position'] = 0.0
    else:
        if all([i < label_element_idx for i in in_idx]):
            score_dict['position'] = 3.0
        elif all([i <= label_element_idx for i in in_idx]):
            score_dict['position'] = 1.0
        else:
            score_dict['position'] = 0.0

    score_dict['total_score'] = sum(value for key, value in score_dict.items() if key != 'total_score') / 5.0
    score_dict['position'] /= 3.0

    return score_dict


if __name__ == '__main__':
    pass
