import torch
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
import heapq

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_purning_grpo_f_ovl_advantage( # grpo adv + purning
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6,  purning: bool = False, purning_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    if isinstance(index,list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2overlenth = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}
    

    bsz = scores.shape[0]
    for i in range(bsz):
        # 检查超长样本，超长样本的reward一般为0，且eos_mask[-1] = 1
        if eos_mask[i][-1] == 1 and scores[i] == 0:
            id2overlenth[index[i]].append(True)
        else:
            id2overlenth[index[i]].append(False) 

        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        unbiased_score = []
        for i in range(len(id2score[idx])):
            if not id2overlenth[idx][i]:
                unbiased_score.append(id2score[idx][i])
        if len(unbiased_score) <= 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(unbiased_score) > 1:
            id2mean[idx] = torch.mean(torch.tensor(unbiased_score))
            id2std[idx] = torch.std(torch.tensor([unbiased_score]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    if Purning: # default purning ratio 0.5
        id2adv = defaultdict(list)
    for i in range(bsz):
        if eos_mask[i][-1] == 1 and scores[i] == 0:
            scores[i] = 0.0
        else:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

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