# Run accelerate accelerate config before
# You may need assign the model path and data path manually.
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=swag
export OOD_TASK=hellaswag
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base

for SEED in 13 21 42 87 100
do
for mixout_p in 0.9
do
  # Train
  accelerate launch run_mc_Mixout.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $TASK_NAME \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --weight_decay 0.1 \
    --do_train \
    --eval_split $EVAL_SPLIT \
    --seed $SEED \
    --mixout_prob $mixout_p \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_Mixout_p=${mixout_p}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_Mixout_p=${mixout_p}_seed=$SEED

  python run_mc_Mixout.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $OOD_TASK \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --eval_split ${EVAL_SPLIT} \
    --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_Mixout_p=${mixout_p}_seed=$SEED \
    --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_Mixout_p=${mixout_p}_seed=$SEED
done
done