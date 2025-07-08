set -x

MODEL_PATH=/workspace/images-ks3-starfs-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
DATA_PATH=/workspace/images-ks3-starfs-hd/dataset/omni_vlr/omni_grpo/geometry3k 
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
export CUDA_VISIBLE_DEVICES=5,6
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_PATH}/data/train-00000-of-00001.parquet \
    data.val_files=${DATA_PATH}/data/validation-00000-of-00001.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2