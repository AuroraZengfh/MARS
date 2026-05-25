MODEL_PATH=$1
QUESTION_PATH=$2
OUTPUT_PATH=$3

python evaluation/qwen_inference.py --model_path ${MODEL_PATH} --question_path ${QUESTION_PATH} --output_path ${OUTPUT_PATH}

