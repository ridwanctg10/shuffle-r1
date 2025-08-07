# Shuffle-R1: Efficient RL framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle

Official code repository of **Shuffle-R1**.

## Introduction
Shuffle-R1 is a a simple yet principled framework that improves RL fine-tuning efficiency by dynamically restructuring trajectory sampling and batch composition. It introduces two key modules:

- **Pairwise Trajectory Sampling (PTS)**
- **Advantage-based Batch Shuffle (ABS)**

Experiments across multiple reasoning benchmarks demonstrate that our framework consistently outperforms strong RL baselines with minimal computational overhead. Specifically, Shuffle-R1 achieves superior performance against GRPO while using only half of the training steps under same settings.

**TL;DR:** We propose Shuffle-R1, a simple and effective RL post-training framework for MLLM that significantly improves RL training efficiency and model performance.

## Release
 - [x] model checkpoints (3B and 7B)
 - [x] datasets
 - [x] training scripts
 - [x] inference scripts via Transformers and vLLM
 - [ ] evaluation scripts


## Framework Overview
![Framework Overview](assets/framework.png)

## Performance Overview
| Model | MathVerse | MathVision | MathVista (mini) | WeMath (loose) | HallusionBench | ChartQA | Avg. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B | 34.8 | 21.9 | 58.4 | 51.7 | 59.8 | 73.1 | 49.9 |
| Qwen2.5-VL-7B | 42.6 | 25.8 | 67.4 | 63.5 | 65.2 | 79.8 | 57.4 |
| Shuffle-R1-3B | 44.2 | 26.8 | 70.4 | 66.5 | 69.2 | 79.9 | 59.5 |
| Shuffle-R1-7B | 53.9 | 30.0 | 77.0 | 72.3 | 71.0 | 84.1 | 64.7 |

All models are evaluated under CoT prompt.

## Try our model

Coming soon.


## Install
Our code is based on [EasyR1](https://github.com/hiyouga/EasyR1). Our code follows a non-intrusive design, which keeps the original functions of EasyR1 unchanged. 

For environment installation, you can: 
 - Refer to [**official instructions**](https://verl.readthedocs.io/en/latest/start/install.html).
 - Or using the [**Dockerfile**](Dockerfile) to build the environment.
 - Or directly using the [**pre-built docker image**](https://hub.docker.com/r/hiyouga/verl).


## Training
### Dataset Format
Supported dataset format is the same as EasyR1. Refer to [**here**](https://github.com/hiyouga/EasyR1?tab=readme-ov-file#custom-dataset) for more information.

### Training Script
```
bash examples/qwen2_5_vl_3b.sh  # 3B model training
bash examples/qwen2_5_vl_7b.sh  # 7B model training
```

## Evaluation
Coming soon.


## Acknowledgement


## Citation