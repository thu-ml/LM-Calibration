import sys
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Mapping, Union
from itertools import chain
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch

class DataCollatorForMultiTask(DataCollatorForLanguageModeling):

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged


        # Here we just mask all
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # Classification label
        if "label" in batch:
            batch["labels_cls"] = batch["label"]
            del batch["label"]
        if "labels" in batch:
            batch["labels_cls"] = batch["labels"]
            del batch["labels"]
        if "label_ids" in batch:
            batch["labels_cls"] = batch["label_ids"]
            del batch["label_ids"]

        # MLM label
        if self.mlm:
            batch["input_ids"], batch["labels_mlm"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels_mlm"] = labels
        return batch
    
class DataCollactorForMCMultiTask(DataCollatorForMultiTask):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        label_name = "label" if "label" in examples[0].keys() else "labels"
        labels = [example.pop(label_name) for example in examples]
        batch_size = len(examples)
        num_choices = len(examples[0]["input_ids"])
        flattened_examples = [
            [{k: v[i] for k, v in example.items()} for i in range(num_choices)] for example in examples
        ]
        examples = list(chain(*flattened_examples))
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # Classification label
        batch["labels_cls"] = torch.tensor(labels, dtype=torch.int64)

        # MLM label
        if self.mlm:
            batch["input_ids"], batch["labels_mlm"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels_mlm"] = labels
        return batch

class DataCollactorForLanguageModelingMC(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        label_name = "label" if "label" in examples[0].keys() else "labels"
        labels = [example.pop(label_name) for example in examples]
        batch_size = len(examples)
        num_choices = len(examples[0]["input_ids"])
        flattened_examples = [
            [{k: v[i] for k, v in example.items()} for i in range(num_choices)] for example in examples
        ]
        examples = list(chain(*flattened_examples))
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # Classification label
        # if "label" in batch:
        #     batch["labels_cls"] = batch["label"]
        #     del batch["label"]
        # if "labels" in batch:
        #     batch["labels_cls"] = batch["labels"]
        #     del batch["labels"]
        # if "label_ids" in batch:
        #     batch["labels_cls"] = batch["label_ids"]
        #     del batch["label_ids"]

        # MLM label
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch