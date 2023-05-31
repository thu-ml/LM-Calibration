# Run accelerate accelerate config before
# You may need assign the model path and data path manually.
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=swag
export OOD_TASK=hellaswag
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base

for SEED in 13 21 42 87 100
do
  # Train
  accelerate launch run_mc_vanilla.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $TASK_NAME \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --weight_decay 0.1 \
    --do_train \
    --eval_split ${EVAL_SPLIT} \
    --seed $SEED \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_seed=$SEED

  # OOD Eval
  python run_mc_vanilla.py \
    --model_name_or_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_seed=${SEED} \
    --dataset_name $OOD_TASK \
    --max_length 256 \
    --eval_split ${EVAL_SPLIT} \
    --per_device_train_batch_size 32 \
    --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_seed=${SEED}
done
