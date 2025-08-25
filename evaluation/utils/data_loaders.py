import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset

def load_geo3k_dataset(data_path: str) -> List[Dict]:
    """Load Geo3K dataset"""
    data_path = os.path.join(data_path, "geometry3k/test")
    dataset = []
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for folder in tqdm(folders, desc="Loading Geo3K data"):
        folder_path = os.path.join(data_path, folder)
        image_path = os.path.join(folder_path, "img_diagram.png")
        json_path = os.path.join(folder_path, "data.json")
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            continue
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        dataset.append({
            "id": data["id"],
            "image_path": image_path,
            "question": data["annotat_text"],
            "answer": data["choices"][mapping[data["answer"]]],
            "dataset": "geo3k"
        })
    
    return dataset

def load_wemath_dataset(data_path: str) -> List[Dict]:
    """Load WeMath dataset"""
    image_root = os.path.join(data_path, "wemath/images")
    data_path = os.path.join(data_path, "wemath/testmini.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        # Determine the image path
        image_path = os.path.join(image_root, item["image_path"])

        dataset.append({
            "id": item["ID"] + "@" + item["key"],
            "image_path": image_path,
            "question": f"{item['question']}\n\nOptions: {item['option']}",
            "answer": item["answer"],
            "dataset": "wemath"
        })
    
    return dataset

def load_mathvista_dataset(data_path: str) -> List[Dict]:
    """Load MathVista dataset"""
    image_base_dir = os.path.join(data_path, "mathvista")
    data_base_dir = os.path.join(data_path, "mathvista")
    dataset_raw = load_dataset(data_base_dir, split="testmini")
    
    dataset = []
    mapping = {
        "0": "A", "1": "B", "2": "C", "3": "D",
        "4": "E", "5": "F", "6": "G", "7": "H"
    }
    
    for item in dataset_raw:
        if item["question_type"] == "multi_choice":
            idx = item["choices"].index(item["answer"])
            answer = mapping[str(idx)]
        else:
            answer = item["answer"]
        
        dataset.append({
            "id": item.get("pid", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query"],
            "answer": answer,
            "task": item["metadata"]["task"],
            "dataset": "mathvista"
        })
    
    return dataset

def load_mathverse_dataset(data_path: str) -> List[Dict]:
    """Load MathVerse dataset"""
    image_base_dir = os.path.join(data_path, "mathverse/images")
    data_path = os.path.join(data_path, "mathverse/testmini.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        dataset.append({
            "id": item.get("sample_index", ""),
            "image_path": os.path.join(image_base_dir, item["image"]),
            "question": item["query_cot"],
            "question_for_eval": item["question_for_eval"],
            "answer": item["answer"],
            "problem_version": item["problem_version"],
            "dataset": "mathverse"
        })
    
    return dataset

def load_mathvision_dataset(data_path: str) -> List[Dict]:
    """Load MathVision dataset"""
    image_base_dir = os.path.join(data_path, "mathvision/images")
    data_path = os.path.join(data_path, "mathvision/MathVision.tsv")
    df = pd.read_csv(data_path, sep='\t')
    
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "id": row.get("index", ""),
            "image_path": os.path.join(image_base_dir, f"{row['index']}.jpg"),
            "question": row["question"],
            "answer": row["answer"],
            "subject": row.get("category", "unknown"),
            "dataset": "mathvision"
        })
    
    return dataset

def load_hallubench_dataset(data_path: str) -> List[Dict]:
    """Load Hallubench dataset"""
    image_base_dir = os.path.join(data_path, "hallubench/images")
    data_path = os.path.join(data_path, "hallubench/HallusionBench.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = []
    for item in data:
        if not item["filename"]:
            continue
        
        if "?" in item["question"]:
            question = item["question"].split("?")[:-1][0]
        else:
            question = item["question"]
        question += "? You final answer can only be yes or no."
        gt_answer = "yes" if int(item["gt_answer"]) == 1 else "no"
        sid, fid, qid = item["set_id"], item["figure_id"], item["question_id"]
        dataset.append({
            "id": f"{sid}_{fid}_{qid}",
            "image_path": os.path.join(image_base_dir, item["filename"].replace("./", "")),
            "question": question,
            "question_for_eval": question,
            "answer": gt_answer,
            "problem_version": item["subcategory"],
            "dataset": "hallubench"
        })
    
    return dataset

def load_chartqa_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "ChartQA_test")
    dataset_raw = load_dataset(data_path, split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": f"chartqa_{idx}",
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "problem_version": item["type"],
            "dataset": "chartqa"
        })
    
    return dataset

def load_logicvista_dataset(data_path: str) -> List[Dict]:
    image_base_dir = os.path.join(data_path, "LogicVista-main/data/images")
    data_path = os.path.join(data_path, "LogicVista-main/data/dataset.json")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = []
    for idx, item in data.items():
        dataset.append({
            "id": idx,
            "image_path": os.path.join(image_base_dir, item["imagename"]),
            "question": item["question"],
            "answer": item["answer"],
            "dataset": "logicvista"
        })
    
    return dataset


def load_r1_onevision_bench(data_path: str) -> List[Dict]:
    image_base_dir = os.path.join(data_path, "r1_onevision_bench")
    data_path = os.path.join(data_path, "r1_onevision_bench/r1_onevision_bench.json")

    with open(data_path, "r") as f:
        dataset_raw = json.load(f)
    
    dataset = []
    for item in dataset_raw:
        dataset.append({
            "id": item["index"],
            "image_path": os.path.join(image_base_dir, item["image_path"]),
            "question": item["question"].split("Question: ")[-1].strip(),
            "answer": item["answer"],
            "level": item["level"],
            "category": item["category"],
            "dataset": "r1-onevision-bench"
        })

    return dataset


def load_math12k_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "math12k")
    dataset_raw = load_dataset(data_path, split="test")
    
    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": idx,
            "question": item["problem"],
            "answer": item["answer"],
            "dataset": "math12k"
        })
    return dataset

def load_aime24_dataset(data_path: str) -> List[Dict]:
    # only contain 30 samples
    data_path = os.path.join(data_path, "AIME24")
    dataset_raw = load_dataset(data_path, split="train")

    dataset = []
    for item in dataset_raw:
        dataset.append({
            "id": item["ID"],
            "question": item["Problem"],
            "answer": item["Answer"],
            "solution": item["Solution"],
            "dataset": "AIME24"
        })
    return dataset

def load_math500_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "MATH500")
    dataset_raw = load_dataset(data_path, split="test")
    
    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": idx,
            "question": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],
            "dataset": "MATH500"
        })
    return dataset

def load_gsm8k_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "GSM8K")
    dataset_raw = load_dataset(data_path, 'main', split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        raw_solution = item["answer"]
        think_process = raw_solution.split("\n####")[0].strip()
        gt_answer = raw_solution.split("\n####")[-1].strip()
        dataset.append({
            "id": idx,
            "question": item["question"],
            "answer": gt_answer,
            "solution": think_process + "\n" + f"The final answer is \\boxed{{{gt_answer}}}",
            "dataset": "GSM8K"
        })
    return dataset

def load_gpqa_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "GPQA_diamond")
    dataset_raw = load_dataset(data_path, split="test")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        raw_problem = item["problem"]
        question = raw_problem.replace("\n\nPlease write your final answer in the form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}", "")
        dataset.append({
            "id": idx,
            "question": question,
            "answer": item["solution"],
            "dataset": "GPQA_diamond"
        })
    return dataset

def load_olympiadbench_dataset(data_path: str) -> List[Dict]:
    data_path = os.path.join(data_path, "OlympiadBench")
    dataset_raw = load_dataset(data_path, split="train")

    dataset = []
    for idx, item in enumerate(dataset_raw):
        dataset.append({
            "id": item["metadata"]["id"],
            "question": item["question"],
            "answer": item["answer"],
            "solution": item["metadata"]["solution"],
            "dataset": "OlympiadBench"
        })
    return dataset