# Run accelerate accelerate config
export CUDA_VISIBLE_DEVICES=9
export TASK_NAME=swag
export OOD_TASK=hellaswag
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME}


mlm_task=wikitext-103-raw-v1
mlm_prob=0.05
temperature=3
lr=5e-5
ls=0.05
kl=1e-4
for SEED in 13 21 42 87 100
do
    # Train
    accelerate launch run_mc_jl.py \
      --model_name_or_path $MODEL_PATH \
      --dataset_name $TASK_NAME \
      --max_length 256 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --weight_decay 0.1 \
      --num_train_epochs 3 \
      --do_train \
      --checkpointing_steps epoch \
      --seed $SEED \
      --mlm_task $mlm_task \
      --mlm_prob $mlm_prob \
      --label_smoothing $ls \
      --kl_temp $kl \
      --temperature $temperature \
      --eval_split ${EVAL_SPLIT} \
      --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=$SEED \
      --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=$SEED

    # OOD Eval
    python run_mc_jl.py \
      --model_name_or_path $MODEL_PATH \
      --dataset_name $OOD_TASK \
      --max_length 256 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --eval_split ${EVAL_SPLIT} \
      --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} \
      --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} 
    

done
  # Eval on test split
  # for T in $TASK_NAME $OOD_TASK
  # do
  #   python run_mc_jl.py \
  #   --model_name_or_path $MODEL_PATH \
  #   --dataset_name $T \
  #   --eval_split test \
  #   --max_length 256 \
  #   --per_device_train_batch_size 32 \
  #   --per_device_eval_batch_size 32 \
  #   --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} \
  #   --conf_dir ./outputs/conf/$T/test/${MODEL_NAME}_vae_lr=${lr}_ls=${ls}_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} 
  # done