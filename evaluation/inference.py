import argparse
import json
import os
import torch
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from utils.data_loaders import (
    load_geo3k_dataset,
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_mathvision_dataset,
    load_hallubench_dataset,
    load_chartqa_dataset,
    load_logicvista_dataset,
    load_r1_onevision_bench
)
from utils.processing import load_image, prepare_prompts
from tqdm import tqdm
import os
import math
from PIL import Image
import concurrent.futures



def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    
    # Model and runtime parameters
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="Gpu memory utilization")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    
    # Dataset selection
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'")
    
    # Dataset-specific paths
    parser.add_argument("--data-path", type=str, default="NoisyRollout/eval/data", help="")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to evaluate
    datasets_to_eval = args.datasets.split(",") if args.datasets != "all" else [
        "geo3k", "wemath", "mathvista", "mathverse", "mathvision", "hallubench", "chartqa", "logicvista", "r1_onevision_bench"
    ]
    
    # Dictionary to store all samples
    all_samples = {}
    
    # Load datasets based on selection
    for dataset_name in datasets_to_eval:
        if dataset_name == "geo3k":
            all_samples["geo3k"] = load_geo3k_dataset(args.data_path)
            print(f"Loaded {len(all_samples['geo3k'])} samples from Geo3K")
        
        elif dataset_name == "wemath":
            all_samples["wemath"] = load_wemath_dataset(args.data_path)
            print(f"Loaded {len(all_samples['wemath'])} samples from WeMath")
        
        elif dataset_name == "mathvista":
            all_samples["mathvista"] = load_mathvista_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathvista'])} samples from MathVista")
        
        elif dataset_name == "mathverse":
            all_samples["mathverse"] = load_mathverse_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathverse'])} samples from MathVerse")
        
        elif dataset_name == "mathvision":
            all_samples["mathvision"] = load_mathvision_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathvision'])} samples from MathVision")
        
        elif dataset_name == "hallubench":
            all_samples["hallubench"] = load_hallubench_dataset(args.data_path)
            print(f"Loaded {len(all_samples['hallubench'])} samples from HalluBench")

        elif dataset_name == "chartqa":
            all_samples["chartqa"] = load_chartqa_dataset(args.data_path)
            print(f"Loaded {len(all_samples['chartqa'])} samples from ChartQA")

        elif dataset_name == "logicvista":
            all_samples["logicvista"] = load_logicvista_dataset(args.data_path)
            print(f"Loaded {len(all_samples['logicvista'])} samples from LogicVista")

        elif dataset_name == "r1_onevision_bench":
            all_samples["r1_onevision_bench"] = load_r1_onevision_bench(args.data_path)
            print(f"Loaded {len(all_samples['r1_onevision_bench'])} samples from R1 OneVision Bench")
    
    if not all_samples:
        print("No datasets loaded. Please check the paths and dataset names.")
        return
    
    # Initialize model
    print(f"Initializing model from {args.model}")
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=False,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=8192,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,  # 0.5
        # top_p=args.top_p,
        max_tokens=args.max_tokens,  # response length=2048
        # repetition_penalty=args.repetition_penalty,
        # detokenize=False
    )

    # Process in batches
    all_results = {}
    for dataset_name in all_samples.keys():
        all_results[dataset_name] = []
    
    for dataset_name, samples in all_samples.items():
        prompts, metadata = prepare_prompts(dataset_name, samples, args)
        
        outputs = llm.generate(prompts, sampling_params)

        ### save outputs for offline evaluation
        output_data = []
        for output in outputs:
            output_dict = {
                "prompt": output.prompt,
                "generated_text": [o.text for o in output.outputs],
                "logprobs": [o.logprobs for o in output.outputs] if output.outputs[0].logprobs else None,
                "finish_reason": [o.finish_reason for o in output.outputs]
            }
            output_data.append(output_dict)

        output_path = os.path.join(args.output_dir, f"{dataset_name}_outputs.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Generated {len(outputs)} outputs for {dataset_name}")


if __name__ == "__main__":
    main()


