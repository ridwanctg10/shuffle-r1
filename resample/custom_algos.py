import torch
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
import heapq
import numpy as np
import pdb

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

EXPERIENCE_SAMPLING_METHODS = [
    'power', 'softmax', 'log_softmax', 'inverse', 'std', 'buffer'
]

EXPERIENCE_SAMPLING_WEIGHT_TARGET = [
    'sum', 'max', 'mean', 'pos'
]

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_purning_grpo_advantage( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    if Purning: # default purning ratio 0.5
        id2adv = defaultdict(list)
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
        # scores[i] = (scores[i] - id2mean[index[i]])
        if Purning:
            bucket_size = int(id2gps[index[i]] * (1.0 - purning_ratio))
            msg: Tuple[int, float] = (abs(scores[i]), i)
            if len(id2adv[index[i]]) < bucket_size:
                heapq.heappush(id2adv[index[i]], msg)
            else:
                heapq.heappushpop(id2adv[index[i]], msg)
    
    if Purning:
        keep_index = []
        for idx in id2adv:
            for item in id2adv[idx]:
                keep_index.append(item[1])
    else:
        keep_index = None

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, keep_index

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_pairwise_purning_grpo_advantage( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}
    

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})  # store the index of the response in the tuple

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    if Purning:
        pairwise_list = []
        for idx in id2score:
            group_scores = [{"adv": (item['score'] - id2mean[idx]) / (id2std[idx] + eps), "index": item['index']} for item in id2score[idx]]  # calculate advantage
            group_scores.sort(key=lambda x: x["adv"], reverse=True)  # sort by adv

            # print(group_scores)
            
            # pairing
            pairs = []
            assert len(group_scores) % 2 == 0, "Number of responses in a group should be even for pairing"
            left, right = 0, len(group_scores) - 1
            while left < right:
                adv1, index1 = group_scores[left]["adv"], group_scores[left]["index"]
                adv2, index2 = group_scores[right]["adv"], group_scores[right]["index"]
                pairs.append({
                    'pair': [group_scores[left], group_scores[right]], 
                    'adv_sum': abs(adv1) + abs(adv2),
                    'adv_max': max(abs(adv1), abs(adv2)),
                    'adv_mean': (abs(adv1) + abs(adv2)) / 2,
                    'pos_adv': abs(torch.max(adv1, adv2))
                })
                left += 1
                right -= 1
            # pairs.sort(key=lambda x: x["adv_sum"], reverse=True)  # sort needed?
            # keep the top k pairs
            bucket_size = int(len(pairs) * (1.0 - purning_ratio))
            pairs_to_keep = pairs[:bucket_size]
            
            for pair in pairs_to_keep:
                # for item in pair['pair']:
                #     keep_index.append(item["index"])
                # pairwise_list.append([item["index"] for item in pair['pair']])  # store the indices of the responses in the pair

                pairwise_list.append({
                    "pair_index": [item["index"] for item in pair['pair']], 
                    "adv_sum": pair['adv_sum'].item(),
                    "adv_max": pair['adv_max'].item(),
                    "adv_mean": pair['adv_mean'].item(),
                    "pos_adv": pair['pos_adv'].item()
                })
    else:
        pairwise_list = None

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_pairwise_purning_grpo_advantage_adv_shift( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}
    

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})  # store the index of the response in the tuple

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    if Purning:
        pairwise_list = []
        for idx in id2score:
            group_scores = [{"adv": (item['score'] - id2mean[idx]) / (id2std[idx] + eps), "index": item['index']} for item in id2score[idx]]  # calculate advantage
            group_scores.sort(key=lambda x: x["adv"], reverse=True)  # sort by adv

            # print(group_scores)
            
            # pairing
            pairs = []
            assert len(group_scores) % 2 == 0, "Number of responses in a group should be even for pairing"
            left, right = 0, len(group_scores) - 1
            while left < right:
                adv1, index1 = group_scores[left]["adv"], group_scores[left]["index"]
                adv2, index2 = group_scores[right]["adv"], group_scores[right]["index"]
                pairs.append({
                    'pair': [group_scores[left], group_scores[right]], 
                    'adv_sum': abs(adv1) + abs(adv2),
                    'adv_max': max(abs(adv1), abs(adv2)),
                    'adv_mean': (abs(adv1) + abs(adv2)) / 2,
                    'pos_adv': abs(torch.max(adv1, adv2))
                })
                left += 1
                right -= 1
            # pairs.sort(key=lambda x: x["adv_sum"], reverse=True)  # sort needed?
            # keep the top k pairs
            bucket_size = int(len(pairs) * (1.0 - purning_ratio))
            pairs_to_keep = pairs[:bucket_size]

            # hack here to add adv shift
            advs_to_keep = [item['adv'] for pair in pairs_to_keep for item in pair['pair']]
            adv_shift_mean = sum(advs_to_keep) / len(advs_to_keep)
            
            for pair in pairs_to_keep:
                # for item in pair['pair']:
                #     keep_index.append(item["index"])
                # pairwise_list.append([item["index"] for item in pair['pair']])  # store the indices of the responses in the pair

                # hack here to add adv shift
                shifted_adv = [item['adv'] - adv_shift_mean for item in pair['pair']]

                pairwise_list.append({
                    "pair_index": [item["index"] for item in pair['pair']], 
                    "adv_sum": pair['adv_sum'].item(),
                    "adv_max": pair['adv_max'].item(),
                    "adv_mean": pair['adv_mean'].item(),
                    "pos_adv": pair['pos_adv'].item(),
                    "shifted_adv": shifted_adv
                })
    else:
        pairwise_list = None

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_pairwise_purning_ablation( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5, ablation_type='max') -> Tuple[torch.Tensor, torch.Tensor]:
    assert ablation_type in ['max', 'min', 'random']

    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})  # store the index of the response in the tuple

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    
    if Purning:
        keep_index = []
        for idx in id2score:
            group_scores = [{"adv": (item['score'] - id2mean[idx]) / (id2std[idx] + eps), "index": item['index']} for item in id2score[idx]]  # calculate advantage
            group_scores.sort(key=lambda x: x["adv"], reverse=True)  # sort by adv

            bucket_size = int(id2gps[idx] * (1.0 - purning_ratio))
            if ablation_type == 'random':
                selected_index = np.random.choice([item['index'] for item in group_scores], bucket_size, replace=False)
            elif ablation_type == 'max':
                selected_index = [item['index'] for item in group_scores[:bucket_size]]
            elif ablation_type == 'min':
                selected_index = [item['index'] for item in group_scores[-bucket_size:]]
            else:
                raise ValueError(f"invalid ablation type {ablation_type}")
            keep_index.extend(selected_index)
    else:
        keep_index = None

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, keep_index

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_pairwise_purning_f_ovl_grpo_advantage( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}
    

    bsz = scores.shape[0]
    for i in range(bsz):
        overlength = False
        # 检查超长样本，超长样本的reward一般为0，且eos_mask[-1] = 1
        if eos_mask[i][-1] == 1 and scores[i] == 0:
            overlength = True
        id2score[index[i]].append({'score': scores[i], 'index': i, 'overlength': overlength})  # store the index of the response in the tuple

    for idx in id2score:
        unbiased_score = [item['score'] for item in id2score[idx] if not item['overlength']]
        id2gps[idx] = len(id2score[idx])
        if len(unbiased_score) <= 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(unbiased_score) > 1:
            id2mean[idx] = torch.mean(torch.tensor(unbiased_score))
            id2std[idx] = torch.std(torch.tensor(unbiased_score))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    if Purning:
        pairwise_list = []
        for idx in id2score:
            group_scores = [
                {"adv": torch.tensor(0.0) if item['overlength'] else (item['score'] - id2mean[idx]) / (id2std[idx] + eps), 
                 "index": item['index'], 
                 'overlength': item['overlength']} 
                for item in id2score[idx]]  # calculate advantage
            group_scores.sort(key=lambda x: x["adv"], reverse=True)  # sort by adv

            # print(group_scores)
            
            # pairing
            pairs = []
            assert len(group_scores) % 2 == 0, "Number of responses in a group should be even for pairing"
            left, right = 0, len(group_scores) - 1
            while left < right:
                adv1, index1 = group_scores[left]["adv"], group_scores[left]["index"]
                adv2, index2 = group_scores[right]["adv"], group_scores[right]["index"]
                pairs.append({'pair': [group_scores[left], group_scores[right]], 'adv_sum': abs(adv1) + abs(adv2)})
                left += 1
                right -= 1

            # keep the top k pairs
            bucket_size = int(len(pairs) * (1.0 - purning_ratio))
            pairs_to_keep = pairs[:bucket_size]
            for pair in pairs_to_keep:
                pairwise_list.append({"pair_index": [item["index"] for item in pair['pair']], "adv_sum": pair['adv_sum'].item()})
    else:
        pairwise_list = None

    for i in range(bsz):
        if eos_mask[i][-1] == 1 and scores[i] == 0:
            scores[i] = 0.0
        else:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_purning_grpo_random( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})  # store the index of the response in the tuple

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    
    if Purning:
        keep_index = []
        for idx in id2score:
            bucket_size = int(id2gps[idx] * (1.0 - purning_ratio))
            selected_index = np.random.choice([item['index'] for item in id2score[idx]], bucket_size, replace=False)
            keep_index.extend(selected_index)
    else:
        keep_index = None

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, keep_index

def normalize_advantage(advantages: np.ndarray, current_step: int, total_training_step: int, method: str = 'power', p: int = 1, t_max: int = 2, t_min: int = 1):
    assert method in EXPERIENCE_SAMPLING_METHODS
    if method in ['power', 'buffer']:
        advantages = advantages ** p
    elif method == 'softmax':  # expand the probability differnece
        assert t_max >= t_min
        temperature = (t_max - t_min) * np.cos(0.5 * np.pi * current_step / total_training_step) + t_min  # bigger temperature, smaller prob difference
        advantages = np.exp(advantages / temperature)
    elif method == 'log_softmax':  # smooth the probability difference
        assert t_max >= t_min
        temperature = (t_max - t_min) * np.sin(0.5 * np.pi * current_step / total_training_step) + t_min  # bigger temperature, bigger prob difference
        advantages = np.log(1 + advantages / temperature)
    elif method == 'inverse':
        advantages = 1 / (advantages + 1e-6)
    else:
        raise ValueError(f"invalid method {method}")
    probabilities = advantages / advantages.sum()
    return probabilities

def sine_normalize_advantage(advantages: np.ndarray, current_step: int, total_training_step: int, method: str = 'power', p: int = 1, t_max: int = 2, t_min: int = 1):
    assert method in EXPERIENCE_SAMPLING_METHODS
    if method == 'power':
        advantages = advantages ** p
    else:
        raise ValueError(f"invalid method {method}")
    probabilities = advantages / advantages.sum()
    num_candidates = len(probabilities)
    sine_coeff = np.sin(np.pi * current_step / total_training_step)
    probabilities = sine_coeff * probabilities + (1 - sine_coeff) * (1 / num_candidates)
    return probabilities


if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0.0, 3.0, 0.0, 0.0, 0.0, 5.0,], 
                                        [0.0, 3.0, 0.0, 1.0, 0.0, 0.0,],
                                        [0.0, 3.0, 0.0, 0.0, 1.0, 0.0,],
                                        [0.0, 3.0, 0.0, 0.0, 0.0, 1.0,],
                                        [0.0, 0.0, 2.0, 3.0, 0.0, 0.0,],
                                        [0.0, 0.0, 2.0, 0.0, 1.0, 0.0,],
                                        [0.0, 0.0, 2.0, 0.0, 0.0, 1.0,],
                                        [0.0, 0.0, 2.0, 1.0, 0.0, 0.0,]])
    eos_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], 
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0,],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0,],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0,],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0,],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0,]])
    
    index = [[1, 1, 1, 1, 1, 1, 1, 1]]
    advantages, returns, keep_index = compute_pairwise_purning_grpo_advantage(token_level_rewards, eos_mask, index, purning=True, purning_ratio=0.5)
    print("Advantages:", advantages)
    print("Returns:", returns)
    print("Keep Index:", keep_index)