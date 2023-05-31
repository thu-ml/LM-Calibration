# Run accelerate accelerate config before
# You may need assign the model path and data path manually.
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=swag
export OOD_TASK=hellaswag
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base

for DELTA in adapter lora prefix
do
for SEED in 13 21 42 87 100
do
for LEARNING_RATE in 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4
do
  # Train
  accelerate launch run_mc_vanilla.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $TASK_NAME \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs 3 \
    --weight_decay 0.1 \
    --do_train \
    --eval_split ${EVAL_SPLIT} \
    --checkpointing_steps epoch \
    --seed $SEED \
    --delta $DELTA \
    --delta_config ./configs/$TASK_NAME/${MODEL_NAME}_${DELTA}/delta.json \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED

  # OOD Eval
  for epoch in 0 1 2
  do
  python run_mc_vanilla.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $OOD_TASK \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --eval_split ${EVAL_SPLIT} \
    --delta $DELTA \
    --delta_config ./configs/$TASK_NAME/${MODEL_NAME}_${DELTA}/delta.json \
    --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=${SEED}/epoch=${epoch} \
    --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=${SEED}/epoch=${epoch}
  done
done
done
done