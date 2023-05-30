for task in SNLI MNLI QQP TwitterPPDB SWAG HellaSWAG
do
    python ./utils/preprocess.py --task $task
done