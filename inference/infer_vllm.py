import argparse
import json
import os
import math
import torch

from typing import List, Dict, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from PIL import Image


def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path)
        if image.mode == "P":
            image = image.convert("RGBA")  # first handle palette+transparency
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")   # discard alpha channel
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts_old(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        image_path = item.get("image_path")
        if isinstance(image_path, str):
            # Skip if image doesn't exists
            if not os.path.exists(image_path):
                continue
            # Load image
            image = load_image(image_path, args.min_pixels, args.max_pixels)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = None
        
        if image is None:
            continue
        
        # Create prompt
        prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{args.system_prompt} <|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "image", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata


def prepare_prompts(samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing total {len(samples)} prompts"):
        image_path = item.get("image_path")
        if isinstance(image_path, str):
            # Skip if image doesn't exists
            if not os.path.exists(image_path):
                continue
            # Load image
            image = load_image(image_path, args.min_pixels, args.max_pixels)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            image = None
        
        if image is None:
            continue
        
        # Create prompt
        prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{args.system_prompt} <|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": item.get("dataset", None),
            "id": item.get("id", None),
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "image", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    
    # Model and runtime parameters
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file, should be in jsonl format")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="Gpu memory utilization")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
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

    assert args.input_file.endswith('.jsonl'), "input file should be in jsonl format"
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    # format of samples like this
    # samples = [
    #     {'image_path': "path/to/image/1", 'question': "question 1"},
    #     {'image_path': "path/to/image/2", 'question': "question 2"},
    # ]  # support batch inference

    
    prompts, metadata = prepare_prompts(samples, args)  # batch process
    
    outputs = llm.generate(prompts, sampling_params)  # batch inference

    ### save outputs
    output_data = []
    for output in outputs:
        output_dict = {
            "prompt": output.prompt,
            "generated_text": [o.text for o in output.outputs],
            "logprobs": [o.logprobs for o in output.outputs] if output.outputs[0].logprobs else None,
            "finish_reason": [o.finish_reason for o in output.outputs]
        }
        output_data.append(output_dict)

    output_path = os.path.join(args.output_dir, f"vllm_outputs.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Generated {len(outputs)} outputs. Saved to {output_path}")



if __name__ == "__main__":
    main()


