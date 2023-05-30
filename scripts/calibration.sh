for task in qqp
do
python3 calibrate.py \
        --test_path "./outputs/conf/${task}/val/roberta-base_seed=42/res.json" \
        --do_evaluate
done