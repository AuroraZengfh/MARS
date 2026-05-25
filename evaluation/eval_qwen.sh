DATA_PATH=$1
PRED_TYPE=$2
THRESHOLD=$3

python evaluation/eval_qwen.py --data_path ${DATA_PATH} --pred_type ${PRED_TYPE} --miou_threshold ${THRESHOLD}