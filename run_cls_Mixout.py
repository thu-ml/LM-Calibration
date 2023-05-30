# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import copy
import logging
import math
import os
import sys
import json
import random
from itertools import chain

import datasets
from datasets import load_dataset, load_metric, load_from_disk
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from models import Variational, Generator, VAE, DataCollatorForMultiTask, LabelSmoothingLoss, replace_layer_for_mixout, recursive_setattr
from utils import check_dir


logger = logging.getLogger(__name__)


task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "TwitterPPDB": ("question1", "question2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--no_sche", action="store_true"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--conf_dir", type=str, default=None, help="Where to store the output of model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--label_smoothing", type=float, default=-1)
    parser.add_argument("--log_epoch", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    # Mixout
    parser.add_argument("--mixout_prob", type=float, default=0.1)
    parser.add_argument("--avg", action="store_true")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None or args.do_train is False, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_from_disk(f'./data/processed_data/{args.task_name}')
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            try:
                label_list = raw_datasets["train"].features["label"].names
                num_labels = len(label_list)
            except:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = raw_datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if args.ckpt_path is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.ckpt_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    model_type="roberta"
    for name, module in tuple(model.named_modules()):
        if model_type in name:
            recursive_setattr(model, name, replace_layer_for_mixout(module, mixout_prob=args.mixout_prob))
    
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    glue_task = ["mnli", "qqp"]

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None and args.task_name not in glue_task:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    # eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    # eval_dataset = processed_datasets["validation"]
    if args.eval_split == "val":
        eval_dataset = processed_datasets["validation"]
    elif args.eval_split == "test":
        eval_dataset = processed_datasets["test"]
    else:
        raise NotImplementedError()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.label_smoothing == -1:
        loss_cls_masked = CrossEntropyLoss()
    else:
        loss_cls_masked = LabelSmoothingLoss(args.label_smoothing, num_labels)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, loss_cls_masked = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, loss_cls_masked
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    

    lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    # if args.task_name is not None:
    #     metric = load_metric("glue", args.task_name)
    # else:
    metric = load_metric("./metrics/accuracy")
    
    # Train!
    if args.do_train:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()

            for step, batch in enumerate(train_dataloader):
                loss_fct = CrossEntropyLoss()

                # outputs = model(**batch)
                labels = batch.pop("labels", None)
                features = model.roberta(**batch)[0]
                if args.avg:
                    features = torch.mean(features, dim=1, keepdim=True)
                logits = model.classifier(features)

                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if args.no_sche:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    if not args.no_sche:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(f"Loss: {loss.detach().float()}")
                    completed_steps += 1
                    
                if completed_steps >= args.max_train_steps:
                    break
                    
            model.eval()
            output_dicts = []
            for step, batch in enumerate(eval_dataloader):
                labels = batch.pop("labels", None)
                features = model.roberta(**batch)[0]
                if args.avg:
                    features = torch.mean(features, dim=1, keepdim=True)
                logits = model.classifier(features).detach()
                predictions = logits.argmax(dim=-1) if not is_regression else logits.squeeze()

                for j in range(logits.size(0)):
                    probs = F.softmax(logits[j], -1)
                    label = labels
                    output_dict = {
                        'index': args.per_device_train_batch_size * step + j,
                        'true': label[j].item(),
                        'pred': logits[j].argmax().item(),
                        'conf': probs.max().item(),
                        'logits': logits[j].cpu().numpy().tolist(),
                        'probs': probs.cpu().numpy().tolist(),
                    }
                    output_dicts.append(output_dict)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(label),
                )

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")

            # If training seems wrong here, abort.
            if eval_metric['accuracy'] < 0.4:
                sys.exit(0)
            
            # Saving model if log_epoch is true
            # if args.log_epoch:
            #     epoch_output_dir = os.path.join(args.output_dir, f"epoch={epoch}")
            #     epoch_conf_dir = os.path.join(args.conf_dir, f"epoch={epoch}")
            #     check_dir(epoch_output_dir)
            #     check_dir(epoch_conf_dir)
            #     epoch_conf_path = os.path.join(epoch_conf_dir, "res.json")
                
            #     accelerator.wait_for_everyone()
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     torch.save(unwrapped_model.state_dict(), os.path.join(epoch_output_dir, "model.ckpt"))

            #     with open(epoch_conf_path, 'w+') as f:
            #         for i, output_dict in enumerate(output_dicts):
            #             output_dict_str = json.dumps(output_dict)
            #             f.write(f'{output_dict_str}\n')
    
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
            
            check_dir(args.conf_dir)
            output_path = os.path.join(args.conf_dir, 'res.json')
            with open(output_path, 'w+') as f:
                for i, output_dict in enumerate(output_dicts):
                    output_dict_str = json.dumps(output_dict)
                    f.write(f'{output_dict_str}\n')

    # Evaluation, we use one process only.
    output_dicts = []
    if accelerator.is_local_main_process:
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            labels = batch.pop("labels", None)
            features = model.roberta(**batch)[0]
            if args.avg:
                features = torch.mean(features, dim=1, keepdim=True)
            logits = model.classifier(features).detach()
            predictions = logits.argmax(dim=-1) if not is_regression else logits.squeeze()

            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                label = labels
                output_dict = {
                    'index': args.per_device_train_batch_size * step + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(labels),
            )

        eval_metric = metric.compute()
        print(eval_metric)

    check_dir(args.conf_dir)
    output_path = os.path.join(args.conf_dir, 'res.json')
    print(f'writing outputs to \'{output_path}\'')

    with open(output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')


if __name__ == "__main__":
    main()