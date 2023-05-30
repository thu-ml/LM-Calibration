# Run accelerate accelerate config before
# You may need assign the model path and data path manually.
export CUDA_VISIBLE_DEVICES=5
export TASK_NAME=qqp
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base
export SEED=42
export MODEL_PATH=../data/huggingface/models/${MODEL_NAME}

if [ $MODEL_NAME = bert-base-uncased ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

if [ $MODEL_NAME = roberta-base ] || [ $MODEL_NAME = microsoft/deberta-v3-base ]; then
    BATCH_SIZE=32
    LEARNING_RATE=1e-5
    WEIGHT_DECAY=0.1
fi

for TASK_NAME in snli
do
if [ $TASK_NAME = snli ]; then
    OOD_TASK=mnli
fi
if [ $TASK_NAME = qqp ]; then
    OOD_TASK=TwitterPPDB
fi

temperature=0.3
mlm_prob=0.4
kl=1e-5
ls=-1
mlm_task=wikitext-103-raw-v1
mlm_task=$TASK_NAME
for SEED in 13 21 42 87 100
do
#   Train
  accelerate launch run_cls_JL.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --max_length 256 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --num_train_epochs 3 \
  --log_epoch \
  --do_train \
  --eval_split $EVAL_SPLIT \
  --seed $SEED \
  --temperature $temperature \
  --mlm_task $mlm_task \
  --mlm_prob $mlm_prob \
  --kl_temp $kl \
  --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=$SEED \
  --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=$SEED


#   # Eval OOD, not using accelerator here.
  python run_cls_JL.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $OOD_TASK \
  --max_length 256 \
  --per_device_train_batch_size $BATCH_SIZE \
  --eval_split ${EVAL_SPLIT} \
  --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} \
  --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED} 

done

# Eval on test split
# for T in $TASK_NAME $OOD_TASK
# do
#     python run_vae_cls_old.py \
#     --model_name_or_path $MODEL_PATH \
#     --task_name $T \
#     --eval_split test \
#     --max_length 256 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED}  \
#     --conf_dir ./outputs/conf/$T/test/${MODEL_NAME}_vae_kl=${kl}_temp=${temperature}_mlm=${mlm_task}-${mlm_prob}_seed=${SEED}  
# done