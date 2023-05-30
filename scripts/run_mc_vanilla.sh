# Run accelerate accelerate config
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=swag
export OOD_TASK=hellaswag
export MODEL_NAME=roberta-base
export EVAL_SPLIT=val
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME}

for SEED in 13 21 42 87 100
# for SEED in 42
do
  # Train
  accelerate launch run_mc_vanilla.py \
    --model_name_or_path $MODEL_PATH \
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

  # for T in $TASK_NAME $OOD_TASK
  # do
  #   python run_mc_vanilla.py \
  #   --model_name_or_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_seed=${SEED} \
  #   --task_name $T \
  #   --max_length 256 \
  #   --eval_split test \
  #   --per_device_train_batch_size $BATCH_SIZE \
  #   --conf_dir ./outputs/conf/$T/test/${MODEL_NAME}_seed=${SEED}
  # done
