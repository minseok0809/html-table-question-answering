# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import json
import os
import collections
import time
import torch
import pandas as pd
from IPython.display import display
from glob import glob
from tqdm import tqdm
from io import StringIO
from transformers.utils import logging
from torch.utils.data import TensorDataset
from processors.mrc_components import MRCExample, MRCInputFeatures, MRCOutputFeatures
from processors.model_io import get_cached_features_file

logger = logging.get_logger(__name__)

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def find(ch, str):
    for i, ltr in enumerate(str):
        if ltr == ch:
            yield i

def get_model_output_feature(encoding_features, model_outputs, overflow_to_sample_mapping):
    mrc_output_features = collections.defaultdict(list)
    tmp_features_dict = [{'tokens': e.tokens,
                          'word_ids': e.word_ids,
                          'offsets': e.offsets}
                         for e in encoding_features]

    start_logits = model_outputs[0]
    end_logits = model_outputs[1]

    for i in range(len(overflow_to_sample_mapping)):
        example_idx = overflow_to_sample_mapping[i]
        current_model_output = (start_logits[i], end_logits[i])
        mrc_output_features[example_idx].append(MRCOutputFeatures(tmp_features_dict[i], current_model_output))

    return mrc_output_features


def get_mrc_input_data_cache(args, tokenizer, phase, target_file_path):
    cached_features_file = get_cached_features_file(args, target_file_path)
    target_file = target_file_path.split("/")[-1]
    # apc: answer_position_character
    if phase == 'train':
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            input_data_cache = torch.load(cached_features_file)
        else:
            input_feature_generation_start = time.time()
            logger.info("Creating features from dataset file at %s %s", args.train_data_dir, target_file)

            examples, excluded_qid_out_of_range = _create_examples(args.train_data_dir, phase, filename=target_file)

            mrc_input_dataset = convert_examples_to_features(examples=examples, tokenizer=tokenizer, args=args,
                                                             is_training=True)

            mrc_input_dataset = TensorDataset(*mrc_input_dataset)

            input_feature_generation_end = time.time()
            logger.info(" generate input feature takes {} s".format(
                input_feature_generation_end - input_feature_generation_start))

            logger.info("Saving features into cached file %s", cached_features_file)
            input_data_cache = {"dataset": mrc_input_dataset}
            torch.save(input_data_cache, cached_features_file)

        return input_data_cache
    else:
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            input_data_cache = torch.load(cached_features_file)
        else:
            input_feature_generation_start = time.time()
            examples, excluded_qid_out_of_range = _create_examples(args.train_data_dir, phase, filename=target_file)

            examples, mrc_input_dataset, mrc_features = convert_examples_to_features(
                examples=examples, tokenizer=tokenizer, args=args, is_training=False
            )

            input_feature_generation_end = time.time()
            logger.info(" generate input feature takes {} s".format(
                input_feature_generation_end - input_feature_generation_start))

            logger.info("Saving features into cached file %s", cached_features_file)
            input_data_cache = {"features": mrc_features, "dataset": mrc_input_dataset, "examples": examples}
            torch.save(input_data_cache, cached_features_file)

        return input_data_cache


def _create_examples(data_dir, phase, filename=None):
    """
    Returns the training examples from the data directory.

    Args:
        data_dir: Directory containing the data files used for training and evaluating.
        phase: "train" or "eval"
        filename: squad form json file

    """
   
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as j:
        input_data = json.load(j)["data"]

    with_numeric_data_qas_id = []
    without_numeric_data_qas_id = []

    excluded_qid = []
    is_training = phase == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["doc_title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]        
            html_wrapped = StringIO(context_text)
            pd.set_option("future.no_silent_downcasting", True)
            source_html_df = pd.read_html(html_wrapped, encoding='utf-8', thousands=None)[0]   
            source_html_df = source_html_df.fillna("na")
            source_html_df.columns = source_html_df.iloc[0]
            source_html_df = source_html_df.iloc[1:, :]
 
            try:
                del source_html_df['비고']
            except:
                pass
            try:
                del source_html_df['순서']
            except:
                pass 
            try:
                del source_html_df['순번']
            except:
                pass
            try:
                del source_html_df['연번']
            except:
                pass
            try:
                del source_html_df['번호']
            except:
                pass
            if source_html_df.columns[0] == source_html_df.iloc[0, 0]:
                # main_columns = list(source_html_df.columns)
                # sub_columns = list(source_html_df.iloc[0, :])
                source_html_df = source_html_df.iloc[1:, :]
                dataframe_column_dim_token = "<2D>"

            elif source_html_df.columns[0] != source_html_df.iloc[0, 0]:
                # main_columns = list(source_html_df.columns)
                dataframe_column_dim_token = "<1D>"

            column_names = []
            for i in range(len(source_html_df.columns)):
                column_names.append(str(i))
            type_token_list = []
            source_html_df.columns = column_names
            for col in source_html_df.columns:
                
                if_number = "Start"
                if "<WND>" not in type_token_list:
                    try:
                        source_html_df[col] = source_html_df[col].astype('int')
                        type_token_list.append("<WND>") 
                    except:
                        pass  
                    for i in source_html_df[col]:
                        if if_number != "Stop":
                            if "<WND>" not in type_token_list:
                                if type(i) != float:
                                    text = i.replace(" " , "")
                                    total_length = len(text)
                                    number_length = len(re.sub(r"[^0-9~:;'[@^{%(-*|,&<`}._=!>;?#$)/±\s]", "", text))
                                    if number_length == 0:
                                        if_number = "Stop"
                                    elif number_length > 0:
                                        number_ratio = number_length / total_length
                                        if number_ratio >= 1.0:
                                            type_token_list.append("<WND>")
                                elif type(i) == float:
                                    if_number = "Stop"
                                    type_token_list.append("<WND>")
                        if if_number == "Stop":
                            pass
                elif "<WND>" in type_token_list:
                    pass

            if "<WND>" in type_token_list:
                dataframe_column_type_token = "<WND>"
            elif "<WND>" not in type_token_list:
                dataframe_column_type_token = "<WOND>"

            revised_context_text = dataframe_column_type_token + " " + context_text

            table_info = _get_table_position_info(revised_context_text)

            for qa in paragraph["qas"]:
                qas_id = qa["question_id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    if is_training:
                        answer = qa["answer"]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"] + len(dataframe_column_type_token) + 1
                    else:
                        answers = qa["answer"]

                try:
                    example = MRCExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=revised_context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        table_info=table_info,
                        answers=answers,
                        is_impossible=is_impossible
                    )
                    examples.append(example)
                except IndexError:
                    excluded_qid.append(qas_id)
                    continue

    return examples, excluded_qid



def _binary_token_search(offsets, ch_position, low=None, high=None):
    if offsets[0][0] > ch_position or max(offsets)[1] < ch_position:
        return -1
    if low is None:
        low = 0
    if high is None:
        high = len(offsets) - 1

    if low > high:
        return -1
    mid = (low + high) // 2
    if offsets[mid][0] > ch_position:
        return _binary_token_search(offsets, ch_position, low, mid - 1)
    if offsets[mid][0] <= ch_position <= offsets[mid][1] - 1:
        return mid
    if offsets[mid][1] - 1 < ch_position:
        return _binary_token_search(offsets, ch_position, mid + 1, high)


def _get_table_position_info(context):
    table_starts = [m.start() for m in re.finditer(r'<table', context)]
    table_ends = [m.start() + len('</table>') for m in re.finditer(r'</table>', context)]
    assert len(table_starts) == len(table_ends), 'Table token does not match'

    all_table_tag_se = [(m.start(), m.end() - 1) for m in re.finditer(
        r'<table.*?>|<thead.*?>|<th.*?>|<td.*?>|<tr.*?>|<tbody.*?>|<tfoot.*?>|<caption.*?>|<colgroup.*?>|<col.*?>|'
        r'</table>|</thead>|</th>|</td>|</tr>|</tbody>|</tfoot>|</caption>|</colgroup>|</col>|'
        r'<p>|</p>',
        context)]

    table_position_info = list(zip(table_starts, table_ends))
    table_info_dict = {
        'table_only': table_position_info,
        'all_table_tags': all_table_tag_se
    }
    return table_info_dict

def get_mrc_input_split_data_example(args, tokenizer, phase, target_file_path):
    target_file = target_file_path.split("/")[-1]
    # apc: answer_position_character
    if phase == 'train':
        input_feature_generation_start = time.time()
        logger.info("Creating features from dataset file at %s %s", args.train_data_dir, target_file)

        examples, excluded_qid_out_of_range,\
            with_numeric_data_qas_id, without_numeric_data_qas_id = _create_examples(args.train_data_dir, phase, filename=target_file)

        examples_list, passages_list, truncated_queries_list = split_examples(examples=examples, tokenizer=tokenizer, args=args,
                                                            is_training=True)
            
        return examples_list, passages_list, truncated_queries_list, with_numeric_data_qas_id, without_numeric_data_qas_id

def split_examples(examples, tokenizer, args, is_training):
    passages = [example.context_text for example in examples]

    truncated_queries = [tokenizer(
        example.question_text, add_special_tokens=False, truncation=True, max_length=args['max_query_length']
    ).encodings[0].tokens for example in examples]
    truncated_queries = [tokenizer.convert_tokens_to_string(truncated_query) for truncated_query in
                         truncated_queries]
    
    len_passages = len(passages) 
    if len(str(len_passages)) == 8: round_num = 5
    elif len(str(len_passages)) == 7: round_num = 4
    elif len(str(len_passages)) == 6: round_num = 3
    elif len(str(len_passages)) == 5: round_num = 2
    elif len(str(len_passages)) == 4: round_num = 1

    index_list = []
    for i in range(1, args.examples_split):
        index_list.append(i * round(len_passages // args.examples_split, round_num))
    index_list.append(len_passages)
    
    examples_list = []
    passages_list = []; truncated_queries_list = []
    for i in range(len(index_list)):
        first_idx = index_list[i] - index_list[0]
        second_idx = index_list[i]
        examples_list.append(examples[first_idx:second_idx])
        passages_list.append(passages[first_idx:second_idx])
        truncated_queries_list.append(truncated_queries[first_idx:second_idx])

    return examples_list, passages_list, truncated_queries_list


def convert_split_examples_to_features(examples, passages, truncated_queries,  tokenizer, args, is_training):

    tokenizer_result = tokenizer(
        truncated_queries,
        passages,
        truncation='only_second',
        padding='max_length',
        max_length=512,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
        stride=args['doc_stride']
    )

    answer_position_character = [(example.start_position_character, example.end_position_character) for example in
                                 examples]
    overflow_to_sample_mapping = tokenizer_result.data['overflow_to_sample_mapping']
    sep_token_idx = [[i for i, e in enumerate(input_ids) if e == 1] for input_ids in
                     tokenizer_result.data['input_ids']]
    sep_token_idx = [[t[1], t[2]] for t in sep_token_idx]
    contexts_offset = [encoding.offsets[sep_idx[0] + 1:sep_idx[1]] for sep_idx, encoding in
                       zip(sep_token_idx, tokenizer_result.encodings)]

    if is_training:
        start_answer_position = [answer_position_character[example_idx][0] for example_idx in
                                 overflow_to_sample_mapping]
        end_answer_position = [answer_position_character[example_idx][1] for example_idx in
                               overflow_to_sample_mapping]

        answer_start_token = list(map(_binary_token_search, contexts_offset, start_answer_position))
        answer_start_token = [at + s[0] + 1 if at != -1 else 0 for at, s in zip(answer_start_token, sep_token_idx)]
        answer_end_token = list(map(_binary_token_search, contexts_offset, end_answer_position))
        answer_end_token = [at + s[0] + 1 if at != -1 else 0 for at, s in zip(answer_end_token, sep_token_idx)]

        mrc_input_dataset = [
            tokenizer_result.data['input_ids'],
            tokenizer_result.data['attention_mask'],
            torch.tensor(answer_start_token),
            torch.tensor(answer_end_token),
        ]

        return mrc_input_dataset

    else:
        mrc_input_dataset = [
            tokenizer_result.data['input_ids'],
            tokenizer_result.data['attention_mask'],
            overflow_to_sample_mapping
        ]
        mrc_features = [
            MRCInputFeatures(e.ids, e.attention_mask, overflow_to_sample_mapping[i].item(), e.tokens, e.word_ids,
                             e.offsets) for i, e in enumerate(tokenizer_result.encodings)]

        mrc_input_dataset = TensorDataset(*mrc_input_dataset)

        return examples, mrc_input_dataset, mrc_features


def convert_examples_to_features(examples, tokenizer, args, is_training):
    passages = [example.context_text for example in examples]

    truncated_queries = [tokenizer(
        example.question_text, add_special_tokens=False, truncation=True, max_length=args['max_query_length']
    ).encodings[0].tokens for example in examples]
    truncated_queries = [tokenizer.convert_tokens_to_string(truncated_query) for truncated_query in
                         truncated_queries]

    tokenizer_result = tokenizer(
        truncated_queries,
        passages,
        truncation='only_second',
        padding='max_length',
        max_length=512,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors='pt',
        stride=args['doc_stride']
    )

    answer_position_character = [(example.start_position_character, example.end_position_character) for example in
                                 examples]
    overflow_to_sample_mapping = tokenizer_result.data['overflow_to_sample_mapping']
    sep_token_idx = [[i for i, e in enumerate(input_ids) if e == 1] for input_ids in
                     tokenizer_result.data['input_ids']]
    sep_token_idx = [[t[1], t[2]] for t in sep_token_idx]
    contexts_offset = [encoding.offsets[sep_idx[0] + 1:sep_idx[1]] for sep_idx, encoding in
                       zip(sep_token_idx, tokenizer_result.encodings)]

    if is_training:
        start_answer_position = [answer_position_character[example_idx][0] for example_idx in
                                 overflow_to_sample_mapping]
        end_answer_position = [answer_position_character[example_idx][1] for example_idx in
                               overflow_to_sample_mapping]

        answer_start_token = list(map(_binary_token_search, contexts_offset, start_answer_position))
        answer_start_token = [at + s[0] + 1 if at != -1 else 0 for at, s in zip(answer_start_token, sep_token_idx)]
        answer_end_token = list(map(_binary_token_search, contexts_offset, end_answer_position))
        answer_end_token = [at + s[0] + 1 if at != -1 else 0 for at, s in zip(answer_end_token, sep_token_idx)]

        mrc_input_dataset = [
            tokenizer_result.data['input_ids'],
            tokenizer_result.data['attention_mask'],
            torch.tensor(answer_start_token),
            torch.tensor(answer_end_token),
        ]

        return mrc_input_dataset

    else:
        mrc_input_dataset = [
            tokenizer_result.data['input_ids'],
            tokenizer_result.data['attention_mask'],
            overflow_to_sample_mapping
        ]
        mrc_features = [
            MRCInputFeatures(e.ids, e.attention_mask, overflow_to_sample_mapping[i].item(), e.tokens, e.word_ids,
                             e.offsets) for i, e in enumerate(tokenizer_result.encodings)]

        mrc_input_dataset = TensorDataset(*mrc_input_dataset)

        return examples, mrc_input_dataset, mrc_features




