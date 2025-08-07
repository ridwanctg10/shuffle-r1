set -x

TIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME=7B-Shuffle-Reasoner

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
DATA_PATH=/your/data/path
SAVE_PATH=saves/vlm/checkpoints/${EXP_NAME}_${TIME} 

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

epoch=20
gpu_num_per_node=8

python3 -m shuffle_r1.main \
    config=examples/config.yaml \
    data.train_files=${DATA_PATH}@train \
    data.val_files=${DATA_PATH}@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=16 \
    worker.rollout.top_p=0.99 \
    worker.actor.purning_ratio=0.5 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.val_override_config.n=8 \
    worker.actor.model.freeze_vision_tower=true \
    trainer.total_episodes=${epoch} \
    trainer.logger=['console','wandb'] \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${gpu_num_per_node} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.experience_replay=true \
    algorithm.adv_estimator=pairwise_purning \
    algorithm.kl_coef=0.0
