import random
import logging
import torch
import numpy as np
import os
import glob
from transformers import (
    BartConfig,
    BartTokenizerFast,
    BartForQuestionAnswering
)






logger = logging.getLogger(__name__)


CONFIG_CLASSES = {
    "bart": BartConfig,
    "kobart-base": BartConfig
}

TOKENIZER_CLASSES = {
    "bart": BartTokenizerFast,
    "kobart-base": BartTokenizerFast
}

MODEL_FOR_QUESTION_ANSWERING = {
    "bart": BartForQuestionAnswering,
    "kobart-base": BartForQuestionAnswering
}


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, output_dir, model, tokenizer, optimizer, scheduler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    # print()
    # print(model_to_save.config.__dict__.items())
    # print()
    model_to_save.config.__dict__.pop("forced_eos_token_id", None)

    # ValueError: Some non-default generation parameters are set in the model config. These should go into either a) `model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file (https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model) 
    # Non-default generation parameters: {'forced_eos_token_id': 1}

    model_to_save.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    if args.save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)


def load_model(target_epoch, model_dir):
    checkpoints = list(
        os.path.dirname(c)
        for c in sorted(glob.glob(model_dir + "/**/" + "pytorch_model.bin", recursive=True))
    )

    check_dict = {int(checkpoint.split('checkpoint-')[-1]): i for i, checkpoint in enumerate(checkpoints)} 
    sorted_checkpoints = sorted(check_dict)
    sorted_checkpoints_paths = [checkpoints[check_dict[checkpoint]] for checkpoint in sorted_checkpoints]

    # get target checkpoint
    target_index = target_epoch
    try:
        checkpoints = list()
        prefix_i = len(sorted_checkpoints) - 1 if target_index == -1 else target_index
        checkpoints.append((prefix_i, sorted_checkpoints_paths[target_index]))

    except IndexError:
        logging.info("Error]Loading checkpoint - %s", str(sorted_checkpoints_paths))
        exit(-1)

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    return checkpoints[target_index]









