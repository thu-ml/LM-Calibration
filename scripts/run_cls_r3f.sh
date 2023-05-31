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
for noise_type in normal
do
for r3f_lambda in 3
do
for r3f_eps in 0.1
do
  # Train
  accelerate launch run_cls_r3f.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --max_length 256 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --do_train \
  --eval_split $EVAL_SPLIT \
  --seed $SEED \
  --noise_type $noise_type \
  --r3f_eps $r3f_eps \
  --r3f_lambda $r3f_lambda \
  --output_dir ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_r3f_noise=${noise_type}_lambda=${r3f_lambda}_eps=${r3f_eps}_seed=$SEED \
  --conf_dir ./outputs/conf/$TASK_NAME/${EVAL_SPLIT}/${MODEL_NAME}_r3f_noise=${noise_type}_lambda=${r3f_lambda}_eps=${r3f_eps}_seed=$SEED


  # Eval OOD, not using accelerator here.
  python run_cls_r3f.py \
  --model_name_or_path ./outputs/ckpts/$TASK_NAME/${MODEL_NAME}_r3f_noise=${noise_type}_lambda=${r3f_lambda}_eps=${r3f_eps}_seed=${SEED} \
  --task_name $OOD_TASK \
  --max_length 256 \
  --eval_split $EVAL_SPLIT \
  --per_device_train_batch_size $BATCH_SIZE \
  --conf_dir ./outputs/conf/$OOD_TASK/${EVAL_SPLIT}/${MODEL_NAME}_r3f_noise=${noise_type}_lambda=${r3f_lambda}_eps=${r3f_eps}_seed=${SEED}
done
done
done
done

done