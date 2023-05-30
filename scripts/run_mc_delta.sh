# Run accelerate accelerate config
export CUDA_VISIBLE_DEVICES=8
export TASK_NAME=swag
export OOD_TASK=hellaswag
export MODEL_NAME=roberta-base
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME} # Pre-downloaded model seems to have some problems...

for DELTA in adapter lora prefix
# for DELTA in adapter
do
# for SEED in 13 21 42 87 100
for SEED in 42
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
    --checkpointing_steps epoch \
    --seed $SEED \
    --delta $DELTA \
    --delta_config ./configs/$TASK_NAME/${MODEL_NAME}_${DELTA}/delta.json \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${MODEL_NAME}_lr=${LEARNING_RATE}_${DELTA}_seed=$SEED

  # OOD Eval
  for epoch in 0 1 2
  do
  python run_mc_vanilla.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $OOD_TASK \
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