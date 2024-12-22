from __future__ import print_function
from collections import Counter
import string
import re
import collections


'''KorQuAD v1.0에 대한 공식 평가 스크립트 '''
'''본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨.'''


def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth and ground_truth == '':
        return 1

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return int((normalize_answer(prediction) == normalize_answer(ground_truth)))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if len(ground_truths) == 0:
        ground_truths.append('')
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_for_score(examples, preds):
    exact_match = dict()
    f1 = dict()

    for example in examples:
        example_id = example.qas_id
        ground_truths = [example.answers['text']] if not example.is_impossible else ['']
        prediction = preds[example_id]

        exact_match[example_id] = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1[example_id] = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    return exact_match, f1


def qa_evaluate(examples, preds):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
    exact, f1 = evaluate_for_score(examples, preds)

    eval_dict = collections.OrderedDict()

    tmp_count_dict = collections.OrderedDict()
    tmp_count_dict['Count_HasAns_exact'] = 0
    tmp_count_dict['Count_HasAns_f1'] = 0
    tmp_count_dict['Count_HasAns_none'] = 0
    tmp_count_dict['Count_HasAns_zero'] = 0
    tmp_count_dict['Count_NoAns_exact'] = 0
    tmp_count_dict['Count_NoAns_zero'] = 0

    for has_ans_example in has_answer_qids:
        if exact[has_ans_example] == 1:
            tmp_count_dict['Count_HasAns_exact'] += 1
        else:
            if preds[has_ans_example] == '':
                tmp_count_dict['Count_HasAns_none'] += 1
            elif f1[has_ans_example] == 0:
                tmp_count_dict['Count_HasAns_zero'] += 1
            else:
                tmp_count_dict['Count_HasAns_f1'] += 1
    for no_ans_example in no_answer_qids:
        if exact[no_ans_example] == 1:
            tmp_count_dict['Count_NoAns_exact'] += 1
        else:
            tmp_count_dict['Count_NoAns_zero'] += 1

    eval_dict.update(tmp_count_dict)

    total = len(has_answer_qids) + len(no_answer_qids)
    tmp_eval_dict = collections.OrderedDict()
    tmp_eval_dict['HasAns_exact'] = 100.0 * sum(exact[k] for k in has_answer_qids) / len(has_answer_qids)
    tmp_eval_dict['HasAns_f1'] = 100.0 * sum(f1[k] for k in has_answer_qids) / len(has_answer_qids)
    tmp_eval_dict['HasAns_total'] = len(has_answer_qids)
    if no_answer_qids:
        tmp_eval_dict['NoAns_exact'] = 100.0 * sum(exact[k] for k in no_answer_qids) / len(no_answer_qids)
        tmp_eval_dict['NoAns_f1'] = 100.0 * sum(f1[k] for k in no_answer_qids) / len(no_answer_qids)
        tmp_eval_dict['NoAns_total'] = len(no_answer_qids)
    tmp_eval_dict['exact'] = 100.0 * sum(exact.values()) / total
    tmp_eval_dict['f1'] = 100.0 * sum(f1.values()) / total
    tmp_eval_dict['total'] = len(examples)

    eval_dict.update(tmp_eval_dict)

    return eval_dict
