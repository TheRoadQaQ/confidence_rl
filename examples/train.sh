set -x

export MODEL_PATH="Qwen2.5-3B-Instruct"
export EXP_NAME=qwen2.5-3b-hotpot-split-adv
export WANDB_PROJECT="new_confidence_exps"

adv_estimator=grpo

# KL settings (same as original DAPO)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False 
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=4096
max_response_length=1536

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Adjusted batch sizes for single node 8 GPUs
train_prompt_bsz=128
gen_prompt_bsz=128
val_prompt_bsz=2048
n_resp_per_prompt=8
train_prompt_mini_bsz=128
filter_groups_enable=False
filter_groups_metric=acc
max_num_gen_batches=5

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.7
val_top_p=1.0
val_top_k=-1

# Performance Related Parameter - Adjusted for single node
sp_size=1
actor_ppo_max_token_len=16384
infer_ppo_max_token_len=16384
offload=False
fsdp_size=-1

train=./dataset/train_dataset/hotpot_train.parquet
test=./dataset/test_dataset/hotpot_test.parquet

train_files="['$train']"
test_files="['$test']"

#actor_rollout_ref.rollout.enforce_eager=False \
#actor_rollout_ref.rollout.free_cache_engine=False \
#actor_rollout_ref.rollout.enable_chunked_prefill=True \

python -m recipe.split_adv_confidence_rl.main_dapo \
    +algorithm.group_confidence_adv_by_acc=True \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${filter_groups_enable} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_prompt_mini_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    custom_reward_function.path="./eval/reward.py" \
    custom_reward_function.name="math_reward_func" \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_before_train=True \
    trainer.test_freq=25 \
    trainer.save_freq=25 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=205 \
    trainer.default_local_dir=./train_results/${WANDB_PROJECT}/${EXP_NAME}