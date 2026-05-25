MODEL_PATH=$1

python scripts/model_merger.py \
    --backend fsdp \
    --local_dir $MODEL_PATH \
    --hf_model_path $MODEL_PATH/huggingface

cd $MODEL_PATH
rm *.pt


