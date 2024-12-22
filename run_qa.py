# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
from transformers.models.bart import BartForQuestionAnswering
import matplotlib.pyplot as plt
import argparse
import json

import logging
import os

import torch
from torch.utils.data import TensorDataset

from attrdict import AttrDict
from utils import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_QUESTION_ANSWERING,
    init_logger,
    set_seed,
    load_model
)
from processors import get_mrc_input_data_cache
import torch.nn as nn
from train_eval import train, evaluate


logger = logging.getLogger(__name__)


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        model_max_length=args.max_seq_length
    )
    ''' add vocab '''
    target_vocab = []
    # target_vocab = {}
    for vf in args.vocab_files:
        with open(os.path.join(args.vocab_dir, vf)) as f:
            lines = f.read().splitlines()
            target_vocab.extend(lines)
            # for line in lines:
            #   target_vocab[line] = line
    # print(target_vocab)            
    # tokenizer.add_special_tokens({"additional_special_tokens": ['<hl>']})
    tokenizer.add_tokens(target_vocab)
    tokenizer.save_pretrained("./")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        model_name_or_path = args.model_name_or_path

        logger.info(" model_name_or_path = %s", model_name_or_path)
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
        )
        model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
            model_name_or_path,
            config=config
        )
        model.resize_token_embeddings(len(tokenizer))

        if args.device == "cuda" and len(args.cuda_visible_devices.split(",")) > 1:
            model = nn.DataParallel(model)
        model.to(args.device)

        train_dataset_filename_list = args.train_files
        all_dataset_feature_list = []
        for train_dataset_filename in train_dataset_filename_list:
            target_file_path = os.path.join(args.train_data_dir, train_dataset_filename)

            train_data_input_feature_cache = get_mrc_input_data_cache(
                args, tokenizer, phase='train', target_file_path=target_file_path
            )
            train_data_input_feature = train_data_input_feature_cache['dataset']
            all_dataset_feature_list.append(train_data_input_feature)

        all_dataset_feature = torch.utils.data.ConcatDataset(all_dataset_feature_list)
        global_step, tr_loss, train_losses = train(args, all_dataset_feature, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        plt.plot(train_losses, 'r')
        plt.show()

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval:
        test_files = args.test_files
        model_dir = args.model_dir

        prefix_i, checkpoint = load_model(args.target_epoch, model_dir)

        result_dir_name = '{}_{}'.format(args.result_dir_name, prefix_i)

        # Reload the model
        model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(checkpoint)
        model.resize_token_embeddings(len(tokenizer))

        if args.device == "cuda" and len(args.cuda_visible_devices.split(",")) > 1:
            model = nn.DataParallel(model)
        model.to(args.device)

        for test_file in test_files:
            result_name = test_file.split('.json')[0]
            target_file_path = os.path.join(args.test_data_dir, test_file)

            eval_input_feature_cache = get_mrc_input_data_cache(
                args, tokenizer, phase='eval', target_file_path=target_file_path
            )

            result = evaluate(args, model, eval_input_feature_cache, result_dir_name, result_name=result_name)

            result_path = os.path.join(result_dir_name, 'result_' + result_name + ".txt")
            logger.info("***** Official Eval results *****")
            with open(result_path, "w", encoding='utf-8') as f:
                logger.info("****** %s ******", os.path.join(os.path.basename(result_dir_name), result_name))
                f.write("{}\n".format(checkpoint))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    f.write(" {} = {}\n".format(key, str(result[key])))


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)
