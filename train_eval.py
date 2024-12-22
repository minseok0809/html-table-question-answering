import os
import logging
import torch
from processors import get_model_output_feature
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from utils import (
    compute_predictions_logits,
    set_seed,
    qa_evaluate,
    save_model,
)
import timeit


logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    train_sampler = RandomSampler(train_dataset)
    train_batch_size = args.train_batch_size

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    # Added here for reproductibility
    set_seed(args)
    losses = []
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[2],
                "end_positions": batch[3],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if len(args.cuda_visible_devices.split(",")) > 1:
                loss = loss.mean()
            loss.backward()

            tr_loss += loss.item()
            if global_step % args.logging_steps == 1:
                logger.info("*****{} - {}: loss: {} *****".format(str(epoch), str(step), str(loss.item())))
                losses.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and args.save_steps == 0:
                    output_dir = os.path.join(args.model_dir, "checkpoint-{}".format(global_step))
                    save_model(args, output_dir, model, tokenizer, optimizer, scheduler)

            if 0 < args.max_steps < global_step:
                break

        if args.save_steps == 0:
            output_dir = os.path.join(args.model_dir, "checkpoint-{}".format(global_step))
            save_model(args, output_dir, model, tokenizer, optimizer, scheduler)
            logger.info("*****{} - {}: loss: {} *****".format(str(epoch), str(step), str(loss.item())))

        mb.write("Epoch {} done".format(epoch + 1))

        if 0 < args.max_steps < global_step:
            break

    return global_step, tr_loss / global_step, losses


def evaluate(args, model, eval_input_feature_cache, result_dir_name, result_name=None):
    eval_examples = eval_input_feature_cache['examples']
    eval_dataset = eval_input_feature_cache['dataset']
    eval_features = eval_input_feature_cache['features']

    overflow_to_sample_mapping = [f[-1].item() for f in eval_dataset]
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format("{MODEL_NAME_EPOCH}"))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    all_start_logits = []
    all_end_logits = []
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            outputs = model(**inputs)

            all_start_logits.extend(outputs.start_logits.tolist())
            all_end_logits.extend(outputs.end_logits.tolist())

    all_results = [all_start_logits, all_end_logits]
    input_feature_for_final_result = get_model_output_feature(eval_features, all_results, overflow_to_sample_mapping)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(eval_dataset))

    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    # Compute predictions
    output_prediction_file = os.path.join(result_dir_name, "predictions_{}.json".format(result_name))

    predictions = compute_predictions_logits(
        eval_examples,
        input_feature_for_final_result,
        overflow_to_sample_mapping,
        args,
        output_prediction_file
    )

    # Compute the F1 and exact scores.
    prediction_texts = {}
    for p in predictions.keys():
        prediction_texts[p] = predictions[p]['predict_text']

    result = qa_evaluate(eval_examples, prediction_texts)

    return result
