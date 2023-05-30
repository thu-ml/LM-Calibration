"""
Modify from https://github.com/shreydesai/calibration
"""


import argparse
import sys
import csv
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict, ClassLabel
from utils import check_dir

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, help='TASK')
parser.add_argument('--input_dir', type=str, default="./data/calibration_data")
parser.add_argument('--output_dir', type=str, default="./data/processed_data")
args = parser.parse_args()

task_name_mapping = {
    "SNLI": "snli",
    "MNLI": "mnli",
    "QQP": "qqp",
    "TwitterPPDB": "TwitterPPDB",
    "SWAG": "swag",
    "HellaSWAG": "hellaswag",
}


csv.field_size_limit(sys.maxsize)

class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = {
            "premise": [],
            "hypothesis": [],
            "label": []
        }
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[7]
                    sentence2 = row[8]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples["premise"].append(sentence1)
                        samples["hypothesis"].append(sentence2)
                        samples["label"].append(label)
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = {
            "premise": [],
            "hypothesis": [],
            "label": []
        }
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples["premise"].append(sentence1)
                        samples["hypothesis"].append(sentence2)
                        samples["label"].append(label)
                except:
                    pass
        return samples

class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = {
            "question1": [],
            "question2": [],
            "label": []
        }
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples["question1"].append(sentence1)
                        samples["question2"].append(sentence2)
                        samples["label"].append(label)
                except:
                    pass
        return samples

class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = {
            "question1": [],
            "question2": [],
            "label": []
        }
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples["question1"].append(sentence1)
                        samples["question2"].append(sentence2)
                        samples["label"].append(label)
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = {
            "sent1": [],
            "sent2": [],
            "ending0": [],
            "ending1": [],
            "ending2": [],
            "ending3": [],
            "label": [],
        }
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    context = row[4]
                    start_ending = row[5]
                    endings = row[7:11]
                    label = int(row[-1])
                    samples["sent1"].append(context)
                    samples["sent2"].append(start_ending)
                    samples["ending0"].append(endings[0])
                    samples["ending1"].append(endings[1])
                    samples["ending2"].append(endings[2])
                    samples["ending3"].append(endings[3])
                    samples["label"].append(label)
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = {
            "ctx_a": [],
            "ctx_b": [],
            "endings": [],
            "label": [],
        }
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples["ctx_a"].append(context)
                    samples["ctx_b"].append(start_ending)
                    samples["endings"].append(endings)
                    samples["label"].append(label)
                except:
                    pass
        return samples

def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()

processor = select_processor()
train_samples = processor.load_samples(f'{args.input_dir}/{args.task}/train.txt')
val_samples = processor.load_samples(f'{args.input_dir}/{args.task}/dev.txt')
test_samples = processor.load_samples(f'{args.input_dir}/{args.task}/test.txt')

train_dataset = Dataset.from_dict(train_samples)
val_dataset = Dataset.from_dict(val_samples)
test_dataset = Dataset.from_dict(test_samples)

raw_dataset = DatasetDict()
raw_dataset["train"] = train_dataset
raw_dataset["validation"] = val_dataset
raw_dataset["test"] = test_dataset

if args.task in ["SNLI", "MNLI"]:
    raw_dataset["train"] = raw_dataset["train"].cast_column("label", ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None))
    raw_dataset["validation"] = raw_dataset["validation"].cast_column("label", ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None))
    raw_dataset["test"] = raw_dataset["test"].cast_column("label", ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None))
elif args.task in ["SWAG", "HellaSWAG"]:
    raw_dataset["train"] = raw_dataset["train"].cast_column("label", ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))
    raw_dataset["validation"] = raw_dataset["validation"].cast_column("label", ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))
    raw_dataset["test"] = raw_dataset["test"].cast_column("label", ClassLabel(num_classes=4, names=['0', '1', '2', '3'], id=None))

assert len(raw_dataset["train"]) == len(train_samples["label"])
assert len(raw_dataset["validation"]) == len(val_samples["label"])
assert len(raw_dataset["test"]) == len(test_samples["label"])

out_dir = f'{args.output_dir}/{task_name_mapping[args.task]}'
check_dir(out_dir)
raw_dataset.save_to_disk(out_dir)
