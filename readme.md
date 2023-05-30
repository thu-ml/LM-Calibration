# Calibration of Fine-tuned Masked Language Models

Offical codebase for [Preserving Pre-trained Features Helps Calibrate Fine-tuned Language Models](https://openreview.net/forum?id=NI7StoWHJPT).

## Setup Environment
``` bash
# (Optional) create a virtual environment
conda create -n LM-Calibration python=3.8
conda activate LM-Calibration

cd LM-Calibration
git clone git@github.com:rubbybbs/OpenDelta.git

# Install torch (example of cu113)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenDelta
cd OpenDelta
python setup.py develop

# Install other dependencies
cd ..
pip install -r requirements.txt
```

You may need to run `accelerate config` to config accelerate too.

## Prepare Dataset
We use the same dataset and split as [Desai & Durrett (2020)](https://github.com/shreydesai/calibration). You can download and preprocess it using:
``` bash
wget "https://cloud.tsinghua.edu.cn/f/aeac582da1d540f7afac/?dl=1" -O calibration_data.tar.gz
tar zxvf calibration_data.tar.gz -C ./data
sh scripts/preprocess_dataset.sh
```

## Training
See bash files on the `./scripts` folder.

## Evaluation
See `./scripts/calibration.sh`, you may need assign the path of the output file produced by training scripts.


## Cite
```bibtex
@inproceedings{
he2023preserving,
title={Preserving Pre-trained Features Helps Calibrate Fine-tuned Language Models},
author={Guande He and Jianfei Chen and Jun Zhu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=NI7StoWHJPT}
}
```

## Acknowledgements
Thanks to the authors of these repositories for the reference implementations of different fine-tuning and evaluation methods.
* https://github.com/shreydesai/calibration
* https://github.com/thunlp/OpenDelta
* https://github.com/alibaba/AliceMind/tree/main/ChildTuning
* https://github.com/bloodwass/mixout
* https://github.com/asappresearch/revisit-bert-finetuning
* https://github.com/Sanyuan-Chen/RecAdam
* https://github.com/facebookresearch/fairseq/tree/main/examples/rxf
