# Run accelerate accelerate config
export CUDA_VISIBLE_DEVICES=6
export TASK_NAME=swag
export OOD_TASK=hellaswag
export EVAL_SPLIT=test
export MODEL_NAME=roberta-base
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME}

# export t0=1000
# export k=0.2

for SEED in 13 21 42 87 100
do
for w in 1
do
  # Train
  accelerate launch run_mc_pwd.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $TASK_NAME \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --weight_decay 0.1 \
    --do_train \
    --eval_split $EVAL_SPLIT \
    --seed $SEED \
    --recadam_anneal_fun constant \
    --recadam_anneal_w $w \
    --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED \
    --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED

  # OOD Eval
  python run_mc_pwd.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $OOD_TASK \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --eval_split ${EVAL_SPLIT} \
    --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED \
    --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED
done
done
done


# export EVAL_SPLIT=test
# for SEED in 13 21 42 87 100
# do
# for T in $TASK_NAME $OOD_TASK
# do
# for w in 0.3 0.4
# do

#   python run_mc_pwd.py \
#     --model_name_or_path $MODEL_PATH \
#     --dataset_name $T \
#     --max_length 256 \
#     --per_device_train_batch_size 32 \
#     --eval_split ${EVAL_SPLIT} \
#     --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED \
#     --conf_dir ./outputs/conf/$T/${EVAL_SPLIT}/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED

# done
# done
# done