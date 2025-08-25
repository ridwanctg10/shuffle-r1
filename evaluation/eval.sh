# #!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3  # adjust according to your GPU configuration

# ========================
# Inference Stage: 
# Get inference results for all the benchmarks
# ========================

# Set data directory for evaluation
# DATA_DIR="/workspace/images-ks3-starfs-hd/dataset/r1/geo_r1_eval"
DATA_DIR="/path/to/evaluation/data"

# Set the model checkpoints for evaluation
# Support evaluation for multiple checkpoints at the same time, each checkpoints occupy a individual GPU.
HF_MODEL_PATHS=(
  "/path/to/checkpoint/one"
  "/path/to/checkpoint/two"
)

# Set the results directories for each checkpoint
RESULTS_DIRS=(
  "results/of/checkpoint/one"
  "results/of/checkpoint/two"
)

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

for i in "${!HF_MODEL_PATHS[@]}"; do
  HF_MODEL_PATH="${HF_MODEL_PATHS[$i]}"
  RESULTS_DIR="${RESULTS_DIRS[$i]}"
  
  echo "Evaluating model: $HF_MODEL_PATH"
  echo "Results will be saved to: $RESULTS_DIR"
  
  python inference.py \
    --model "$HF_MODEL_PATH" \
    --output-dir "$RESULTS_DIR" \
    --data-path "$DATA_DIR" \
    --datasets geo3k,hallubench,mathverse,mathvision,wemath,mathvista,chartqa \
    --tensor-parallel-size 1 \
    --system-prompt="$SYSTEM_PROMPT" \
    --min-pixels 262144 \
    --max-pixels 4194304 \
    --max-model-len 8192 \
    --temperature 0.5 
  
  echo "Finished evaluating $HF_MODEL_PATH"
  echo "-----------------------------------"
done


# ========================
# Evaluation Stage: 
# Evaluate the inference results for all the benchmarks
# ========================

for i in "${!HF_MODEL_PATHS[@]}"; do
  HF_MODEL_PATH="${HF_MODEL_PATHS[$i]}"
  RESULTS_DIR="${RESULTS_DIRS[$i]}"
  
  echo "Evaluating model: $HF_MODEL_PATH"
  echo "Results will be saved to: $RESULTS_DIR"
  
  python evaluation.py \
    --data-path "$DATA_DIR" \
    --datasets geo3k,hallubench,mathverse,mathvision,wemath,mathvista,chartqa \
    --output-dir "$RESULTS_DIR" \
