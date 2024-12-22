

class MRCExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        table_info,
        answers,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        if start_position_character is not None:
            self.start_position_character = start_position_character
            self.end_position_character = start_position_character + len(answer_text) - 1
        else:
            self.start_position_character = -1
            self.end_position_character = -1

        self.table_info = table_info


class MRCInputFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        example_index: the index of the example
        tokens: list of tokens corresponding to the input ids
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        example_index,
        tokens,
        word_ids,
        offsets
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

        self.example_index = example_index
        self.tokens = tokens
        self.word_ids = word_ids
        self.offsets = offsets


class MRCOutputFeatures:
    def __init__(self, tmp_features, model_output):

        self.tokens = tmp_features['tokens']
        self.word_ids = tmp_features['word_ids']
        self.offsets = tmp_features['offsets']

        self._get_token_to_orig_map()

        self.model_results = MRCResult(model_output[0], model_output[1])

    def _get_token_to_orig_map(self):
        context_start_token_idx = self.tokens.index('</s>') + 2

        word_info = self.word_ids[context_start_token_idx:]
        self.token_to_orig_map = {k + context_start_token_idx: v for k, v in enumerate(word_info[:-1])}

        offset_info = self.offsets[context_start_token_idx:]
        self.token_to_orig_ch_map = {k + context_start_token_idx: v[0] for k, v in enumerate(offset_info[:-1])}


class MRCResult(object):
    def __init__(self, start_logits, end_logits):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.span_null_score = start_logits[0] + end_logits[0]
