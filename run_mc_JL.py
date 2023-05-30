#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import copy
from curses import def_prog_mode
import json
import logging
import math
import os
import sys
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

import datasets
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from datasets import load_dataset, load_metric, load_from_disk, Value, ClassLabel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.file_utils import PaddingStrategy, get_full_repo_name

from models import VariationalForMC, Generator, VAE, DataCollatorForMultiTask, DataCollactorForMCMultiTask, LabelSmoothingLoss
from utils import check_dir


logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--mlm_task",
        type=str,
        default=None,
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
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
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
        default=8,
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
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--conf_dir", type=str, default=None, help="Where to store the final model logits.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        required=False,
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--mlm_len", type=int, default=64)
    parser.add_argument("--mlm_batch_size", type=int, default=8)
    parser.add_argument("--mlm_cls_temp", type=float, default=0.)
    parser.add_argument("--label_smoothing", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--kl_temp", type=float, default=0.)
    parser.add_argument("--cls_temp", type=float, default=1.)
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all") if args.with_tracking else Accelerator()
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

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     check_dir(args.output_dir)
    #     check_dir(args.conf_dir)
    # accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        raw_datasets = load_from_disk(f'./data/processed_data/{args.dataset_name}')
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if args.model_name_or_path:
    #     model = AutoModelForMultipleChoice.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForMultipleChoice.from_config(config)

    model_pretrained = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    model_mc = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        add_pooling_layer=False,
    )
    model_mc.resize_token_embeddings(len(tokenizer))

    variational = VariationalForMC(model_mc)
    generator = Generator(copy.deepcopy(model_pretrained.lm_head), config)
    model = VAE(config, variational, generator)

    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, "model.ckpt")), strict=False)

    # Freeze the pretrained model
    for param in model_pretrained.parameters():
        param.requires_grad = False

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    #NOTE: Only Swag & Hellaswag are supported now.
    if args.dataset_name == 'swag':
        ending_names = [f"ending{i}" for i in range(4)]
        context_name = "sent1"
        question_header_name = "sent2"

        def get_sentences(examples):
            first_sentences = [[context] * 4 for context in examples[context_name]]
            question_headers = examples[question_header_name]
            second_sentences = [
                [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
            ]
            return first_sentences, second_sentences
    elif args.dataset_name =="hellaswag":
        ending_names = "endings"
        context_name = "ctx_a"
        question_header_name = "ctx_b"

        def get_sentences(examples):
            first_sentences = [[context] * 4 for context in examples[context_name]]
            question_headers = examples[question_header_name]
            second_sentences = [
                [f"{header} {end}" for end in examples[ending_names][i]] for i, header in enumerate(question_headers)
            ]
            return first_sentences, second_sentences
    else:
        raise NotImplementedError()

    label_column_name = "label" if "label" in column_names else "labels"
    if args.dataset_name == "hellaswag":
        raw_datasets["train"] = raw_datasets["train"].cast_column(label_column_name, ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))
        raw_datasets["validation"] = raw_datasets["validation"].cast_column(label_column_name, ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))
        raw_datasets["test"] = raw_datasets["test"].cast_column(label_column_name, ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))
    def preprocess_function(examples):
        first_sentences, second_sentences = get_sentences(examples)
        labels = examples[label_column_name]
        
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
        )

    train_dataset = processed_datasets["train"]
    if args.eval_split == "val":
        eval_dataset = processed_datasets["validation"]
    elif args.eval_split == "test":
        eval_dataset = processed_datasets["test"]
    else:
        raise NotImplementedError()

    mlm_flat_flag = True
    if args.mlm_task is not None and args.mlm_task not in ["swag"]:
        mlm_flat_flag = False
        raw_datasets_mlm = load_dataset(args.mlm_task)
        column_names = raw_datasets_mlm["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        max_seq_length = args.mlm_len
        if args.mlm_task == "bookcorpus":
            raw_datasets_mlm = raw_datasets_mlm.filter(lambda example, idx: idx % 5 == 0, with_indices=True, num_proc=8)
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets_mlm.map(
                tokenize_function,
                batched=True,
                num_proc=8,
                remove_columns=column_names,
                desc="Running tokenizer on every text in dataset",
                load_from_cache_file=True,
            )
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        
        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=8,
                desc=f"Grouping texts in chunks of {max_seq_length}",
                load_from_cache_file=True,
            )

        train_dataset_mlm = tokenized_datasets["train"]

    else:
        args.mlm_task = args.dataset_name
        train_dataset_mlm = train_dataset

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
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    if mlm_flat_flag:
        data_collator_mix = DataCollactorForMCMultiTask(tokenizer, mlm_probability=args.mlm_prob)
        train_dataloader_mix = DataLoader(
            train_dataset_mlm, shuffle=True, collate_fn=data_collator_mix, batch_size=args.mlm_batch_size
        )
    else:
        data_collator_mix = DataCollatorForMultiTask(tokenizer, mlm_probability=args.mlm_prob)
        train_dataloader_mix = DataLoader(
            train_dataset_mlm, shuffle=True, collate_fn=data_collator_mix, batch_size=args.mlm_batch_size
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

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

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
    
    if args.label_smoothing == -1:
        loss_cls = CrossEntropyLoss()
    else:
        loss_cls = LabelSmoothingLoss(args.label_smoothing, 4)

    # Prepare everything with our `accelerator`.
    model, model_pretrained, optimizer, train_dataloader, train_dataloader_mix, eval_dataloader, loss_cls = accelerator.prepare(
        model, model_pretrained, optimizer, train_dataloader, train_dataloader_mix, eval_dataloader, loss_cls
    )

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        accelerator.init_trackers("clm_no_trainer", args)

    # Metrics
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

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                resume_step = None
                path = args.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            if "epoch" in path:
                args.num_train_epochs -= int(path.replace("epoch_", ""))
            else:
                resume_step = int(path.replace("step_", ""))
                args.num_train_epochs -= resume_step // len(train_dataloader)
                resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

        
        for epoch in range(args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, (batch, batch_mix) in enumerate(zip(train_dataloader, train_dataloader_mix)):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == 0 and step < resume_step:
                    continue
                labels_cls = batch.pop("labels", None)
                labels_mlm = batch_mix.pop("labels_mlm", None)
                labels_mlm_cls = batch_mix.pop("labels_cls", None)

                loss_mlm = CrossEntropyLoss()

                # MLM + Classification
                logits_mlm_cls, z1 = model.variational(**batch_mix, flat=False)
                logits_px = model.generator(z1)

                # CLS
                logits_cls, z2 = model.variational(**batch)

                # Align with original pretrained LM
                with torch.no_grad():
                    output_pretrained = model_pretrained(**batch_mix, labels=labels_mlm)
                    mask_token_index = (labels_mlm.view(-1) != loss_mlm.ignore_index).nonzero().squeeze(-1)
                    labels_mlm = output_pretrained.logits.view(-1, model.config.vocab_size)[mask_token_index].softmax(-1)
                
                # Classification Error Unmasked
                # loss = loss_mlm(logits_cls, labels_cls)
                loss = args.cls_temp * loss_cls(logits_cls, labels_cls)
                # loss += args.mlm_cls_temp * loss_cls(logits_mlm_cls.view(-1, 4), labels_mlm_cls)

                # ELBO
                reconstruct = args.temperature * loss_mlm(logits_px.view(-1, model.config.vocab_size)[mask_token_index], labels_mlm)
                # reconstruct = args.temperature * loss_mlm(logits_px.view(-1, model.config.vocab_size), labels_mlm.view(-1))
                kl = args.kl_temp * (torch.sum(torch.pow(torch.mean(z1, dim=1), 2)) + torch.sum(torch.pow(torch.mean(z2, dim=1), 2)))
                loss += reconstruct + kl
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(f"Loss: {loss.detach().float()}")
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break
                
            model.eval()
            output_dicts = []
            for step, batch in enumerate(eval_dataloader):
                labels = batch.pop("labels", None)
                with torch.no_grad():
                    logits, _ = model.variational(**batch)
                predictions = logits.argmax(dim=-1)
                logits = logits.detach()
                for j in range(logits.size(0)):
                    probs = F.softmax(logits[j], -1)
                    output_dict = {
                        'index': args.per_device_train_batch_size * step + j,
                        'true': labels[j].item(),
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
            accelerator.print(f"epoch {epoch}: {eval_metric}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": eval_metric,
                        "train_loss": total_loss,
                        "epoch": epoch,
                    },
                    step=completed_steps,
                )

            # if args.checkpointing_steps == "epoch":
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
            check_dir(args.output_dir)
            check_dir(args.conf_dir)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(args.output_dir, "model.ckpt"))
            conf_path = os.path.join(args.conf_dir, 'res.json')
            with open(conf_path, 'w+') as f:
                for i, output_dict in enumerate(output_dicts):
                    output_dict_str = json.dumps(output_dict)
                    f.write(f'{output_dict_str}\n')

    # Evaluation, we use one process only.
    output_dicts = []
    if accelerator.is_local_main_process:
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            labels = batch.pop("labels", None)
            with torch.no_grad():
                logits, _ = model.variational(**batch)
            predictions = logits.argmax(dim=-1)
            logits = logits.detach()
            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.per_device_train_batch_size * step + j,
                    'true': labels[j].item(),
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