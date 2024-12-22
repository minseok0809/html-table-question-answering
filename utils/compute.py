import json
import logging
import collections
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_nbest(prelim_predictions, example, features, args):
    _nbest_prediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_offset", "end_offset", "char_positions"]
    )
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= args['n_best_size']:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction

            answer_ch_start = feature.offsets[pred.start_index][0]
            answer_ch_end = feature.offsets[pred.end_index][1]
            final_text = example.context_text[answer_ch_start:answer_ch_end]
            with open("./a.txt", "a") as f:
                f.write(final_text + "\n")

            if final_text in seen_predictions:
                continue

            # get start_position and end_position from final_text
            seen_predictions[final_text] = True

        else:
            final_text = ""
            answer_ch_start = -1
            answer_ch_end = -1
            seen_predictions[final_text] = True

        if '<t' in final_text or '</t' in final_text:
            continue

        nbest.append(_nbest_prediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
                                       start_offset=pred.start_index, end_offset=pred.end_index,
                                       char_positions=(answer_ch_start, answer_ch_end)))

    return nbest


def _get_prelim_prediction(feature_index, start_index, end_index, start_logit, end_logit):
    _prelim_prediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )
    return _prelim_prediction(
        feature_index=feature_index,
        start_index=start_index,
        end_index=end_index,
        start_logit=start_logit,
        end_logit=end_logit,
    )


def _is_valid_idx_pair(start_index, end_index, feature, max_answer_length, table_tag_info):
    # We could hypothetically create invalid predictions, e.g., predict
    # that the start of the span is in the question. We throw out all
    # invalid predictions.
    if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):
        return False

    if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
        return False

    if end_index < start_index:
        return False

    for table_tag in table_tag_info:
        if table_tag[0] <= feature.token_to_orig_ch_map[start_index] <= table_tag[1] \
                or table_tag[0] <= feature.token_to_orig_ch_map[end_index] <= table_tag[1]:
            return False
    length = end_index - start_index + 1
    if length > max_answer_length:
        return False

    return True


def _get_valid_predictions(indices, feature_index, feature, max_answer_length, table_tag_info,
                           null_score_diff_threshold):
    prelim_predictions = []

    start_indexes = indices['start_indexes']
    end_indexes = indices['end_indexes']
    for start_index in start_indexes:
        for end_index in end_indexes:
            if not _is_valid_idx_pair(start_index, end_index, feature, max_answer_length, table_tag_info):
                continue
            if feature.model_results.start_logits[start_index] < 0 or feature.model_results.end_logits[end_index] < 0:
                continue
            cur_span_logit = feature.model_results.start_logits[start_index] + feature.model_results.end_logits[end_index]
            if cur_span_logit + null_score_diff_threshold < feature.model_results.span_null_score:
                continue

            start_logit = feature.model_results.start_logits[start_index]
            end_logit = feature.model_results.end_logits[end_index]
            prelim_predictions.append(
                _get_prelim_prediction(feature_index, start_index, end_index, start_logit, end_logit)
            )

    prelim_predictions.append(
        _get_prelim_prediction(feature_index, 0, 0,
                               feature.model_results.span_null_score / 2,
                               feature.model_results.span_null_score / 2)
    )

    return prelim_predictions


def _set_result(nbest_dict, nbest, example):
    total_scores = [entry.start_logit + entry.end_logit for entry in nbest]

    probs = _compute_discriminative_raw_score(total_scores)
    nbest_json = []
    for (i, entry) in enumerate(nbest):
        if not entry.text:
            continue
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        output["start_offset"] = entry.start_offset
        output["end_offset"] = entry.end_offset
        output["char_positions"] = entry.char_positions
        nbest_json.append(output)

    if not nbest_json:
        nbest_json.append(
            {
                'text': '',
                'probability': -1
            }
        )

    nbest_dict[example.qas_id] = nbest_json
    return nbest_dict


def compute_predictions_logits(all_examples, all_features, overflow_to_sample_mapping, args, output_prediction_file):

    logger.info(f"Writing predictions to: {output_prediction_file}")

    all_nbest_dict = collections.OrderedDict()

    example_index = list(dict.fromkeys(overflow_to_sample_mapping))

    all_predictions = collections.OrderedDict()
    for e_idx in tqdm(example_index):
        example = all_examples[e_idx]
        features = []
        prelim_predictions = []

        for output_feature_index, output_feature in enumerate(all_features[e_idx]):
            start_indexes = _get_best_indexes(output_feature.model_results.start_logits, args['n_best_size'])
            end_indexes = _get_best_indexes(output_feature.model_results.end_logits, args['n_best_size'])

            indices = {
                'start_indexes': start_indexes,
                'end_indexes': end_indexes
            }
            features.append(output_feature)
            prelim_predictions.extend(
                _get_valid_predictions(indices, output_feature_index, output_feature, args['max_answer_length'],
                                       example.table_info['all_table_tags'], args['null_score_diff_threshold']))

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        nbest = _get_nbest(prelim_predictions, example, features, args)

        all_nbest_dict = _set_result(all_nbest_dict, nbest, example)

        all_predictions[example.qas_id] = dict()
        all_predictions[example.qas_id]['predict_text'] = all_nbest_dict[example.qas_id][0]['text']
        all_predictions[example.qas_id]['score'] = all_nbest_dict[example.qas_id][0]['probability'] * 100

    with open(output_prediction_file, 'w') as j:
        json.dump(all_predictions, j, indent='\t', ensure_ascii=False)

    return all_predictions


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    z = np.array(scores)
    max_z = np.max(z)
    exp_z = np.exp(z-max_z)
    sum_ex_z = np.sum(exp_z)
    y = exp_z / sum_ex_z
    probs = list(y)

    return probs


def _compute_discriminative_raw_score(raw_scores):

    raw_scores = np.array(raw_scores)
    tmp_scaling_factor = 1 if np.sqrt(np.std(raw_scores)) == 0 else np.sqrt(np.std(raw_scores))
    raw_scores = raw_scores/tmp_scaling_factor

    final_scores = _compute_softmax(raw_scores.tolist())

    return final_scores
