# Run accelerate accelerate config before
# You may need assign the model path and data path manually.
export CUDA_VISIBLE_DEVICES=0
export EVAL_SPLIT=val
export MODEL_NAME=roberta-base

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

for TASK_NAME in qqp
do
if [ $TASK_NAME = snli ]; then
    OOD_TASK=mnli
fi
if [ $TASK_NAME = qqp ]; then
    OOD_TASK=TwitterPPDB
fi

for SEED in 13 21 42 87 100
do
for w in 20
do
#   Train
  accelerate launch run_cls_pwd.py \
  --model_name_or_path $MODEL_NAME \
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
  --recadam_anneal_fun constant \
  --recadam_anneal_w $w \
  --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED \
  --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED


  # Eval OOD, not using accelerator here.
  python run_cls_pwd.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $OOD_TASK \
  --max_length 256 \
  --per_device_train_batch_size $BATCH_SIZE \
  --eval_split ${EVAL_SPLIT} \
  --ckpt_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED \
  --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_RecAdam_weight=${w}_seed=$SEED
done
done
done

done


