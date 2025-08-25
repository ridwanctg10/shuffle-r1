import argparse
import json
import os

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
from utils.processing import (
    prepare_prompts,
    process_outputs,
    calculate_metrics
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex.json"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    
    # Model and runtime parameters
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--eval-threads", type=int, default=12, help="Number of threads for evaluation")
    
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
        "geo3k", "wemath", "mathvista", "mathverse", "mathvision", "hallubench", "chartqa", "logicvista"
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


    # Process in batches
    all_results = {}
    for dataset_name in all_samples.keys():
        all_results[dataset_name] = []

    all_metrics = []
    
    for dataset_name, samples in all_samples.items():
        prompts, metadata = prepare_prompts(dataset_name, samples, args)


        generated_path = os.path.join(args.output_dir, f"{dataset_name}_outputs.json")
        
        outputs = json.load(open(generated_path, 'r', encoding='utf-8'))
        
        # Process outputs
        results = process_outputs(outputs, metadata, args.eval_threads)
        all_results[dataset_name] = results
        
        metrics = calculate_metrics(results)
        
        output_dict = {
            "results": results,
            "metrics": metrics,
            "config": vars(args)
        }
        
        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
        print(f"{dataset_name.upper()} Results:")
        print(f"  Total samples: {len(results)}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'sub_accuracies' in metrics:
            print("  Task/Category Accuracies:")
            for task, acc in metrics['sub_accuracies'].items():
                print(f"    {task}: {acc:.4f}")
        print()

        all_metrics.append({dataset_name: metrics})
    
    with open(os.path.join(args.output_dir, "all_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()