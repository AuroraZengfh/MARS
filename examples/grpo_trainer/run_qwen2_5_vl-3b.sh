set -x
ENGINE=${1:-vllm}

ROOT_PATH=/path/to/your/root
MODEL_PATH=/path/to/your/cot/model

DATASET=$2
REWARD_PATH=$3
NUM_IMAGE=$4

PROJ_NAME='GRPO'
EXP_NAME='QWEN2.5_VL_3B-MARS'

loss_agg_mode=token-mean
REWARD_NAME=compute_score

export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${ROOT_PATH}/datasets/${DATASET}/train.parquet \
    data.val_files=${ROOT_PATH}/datasets/${DATASET}/test.parquet \
    data.custom_cls.name=llvip \
    ray_init.num_cpus=16 \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.num_image=${num_image} \
    data.dataloader_num_workers=0 \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_NAME} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=14 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.rollout_data_dir='rollout_data' \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 $@
    