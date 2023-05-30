# Run accelerate accelerate config
export CUDA_VISIBLE_DEVICES=7
export TASK_NAME=qqp
export OOD_TASK=TwitterPPDB
export MODEL_NAME=roberta-base
export SEED=42
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME}

for DELTA in adapter lora
do

if [ $DELTA = lora ]; then
    LEARNING_RATE=2e-4
fi
if [ $DELTA = prefix ]; then
    LEARNING_RATE=1e-4
fi
if [ $DELTA = adapter ]; then
    LEARNING_RATE=2e-4
fi

for TASK_NAME in qqp snli
do
if [ $TASK_NAME = snli ]; then
    OOD_TASK=mnli
fi
if [ $TASK_NAME = qqp ]; then
    OOD_TASK=TwitterPPDB
fi

for SEED in 21 42 87
# for SEED in 42
do
  accelerate launch run_cls_vanilla.py \
    --model_name_or_path $MODEL_PATH \
    --task_name $TASK_NAME \
    --seed $SEED \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs 3 \
    --do_train \
    --log_epoch \
    --delta $DELTA \
    --delta_config ./configs/$TASK_NAME/${MODEL_NAME}_${DELTA}/delta.json \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED

    for epoch in 0 1 2
    do
    python run_cls_vanilla.py \
    --model_name_or_path $MODEL_PATH \
    --task_name $OOD_TASK \
    --seed $SEED \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --delta $DELTA \
    --delta_config ./configs/$TASK_NAME/${MODEL_NAME}_${DELTA}/delta.json \
    --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=${SEED}/epoch=${epoch} \
    --conf_dir ./outputs/conf/$OOD_TASK/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=${SEED}/epoch=${epoch}
    done
done
done
done