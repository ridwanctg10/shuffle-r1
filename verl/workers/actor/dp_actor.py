# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig

import torch.nn.functional as F


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]

def build_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    mask_token_ids = [151652, 151653, 151655]
    # 1. 创建一个与 input_ids 形状相同，且所有元素都为 1 的 attention_mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    # 2. 遍历需要 mask掉的 token ID
    for token_id in mask_token_ids:
        # 找到 input_ids 中所有等于当前 token_id 的位置
        # 并将这些位置在 attention_mask 中置为 0
        attention_mask[input_ids == token_id] = 0
    
    return attention_mask

class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float, remove_images_for_logp: bool = False) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        Args:
            micro_batch: The micro batch of data.
            temperature: Temperature for scaling logits.
            remove_images_for_logp: If True, multi-modal inputs (images) will be removed when computing logp. Defaults to False.
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        # --- 修改开始 ---
        # 根据参数 remove_images_for_logp 决定是否包含多模态输入
        multi_modal_inputs_to_pass = {}
        if not remove_images_for_logp: # 如果 remove_images_for_logp 为 False，则包含多模态输入
            if "multi_modal_inputs" in micro_batch:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs_to_pass[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )
        # --- 修改结束 ---

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            if remove_images_for_logp: # build attention mask for logp mask掉 <|vision_start|><|image_pad|><|vision_end|>
                attention_mask = build_attention_mask(input_ids_rmpad)
            else:
                attention_mask = None

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=attention_mask,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs_to_pass,  # 使用条件化处理后的多模态输入
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            if self.config.ulysses_sequence_parallel_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            # add here to calculate policy entropy
            # chunk_size=1024
            # with torch.no_grad():
            #     results = []
            #     for i in range(0, logits_rmpad.size(0), chunk_size):
            #         chunk_logits = logits_rmpad[i:i+chunk_size]
            #         ent_probs = F.softmax(chunk_logits, dim=-1)
            #         entropy = -torch.sum(ent_probs * torch.log(1e-8 + ent_probs), dim=-1)
            #         results.append(entropy)
            #     # ent_probs = F.softmax(logits_rmpad, dim=-1)
            #     # entropy = -torch.sum(ent_probs * torch.log(1e-8 + ent_probs), dim=-1)

            #     entropy = torch.cat(results, dim=0)
            # # with torch.no_grad():
            # #     current_device = logits_rmpad.device
            # #     with torch.autocast(device_type=current_device.type, dtype=torch.bfloat16):
            # #         ent_probs = F.softmax(logits_rmpad, dim=-1)
            # #         entropy = -torch.sum(ent_probs * torch.log(1e-8 + ent_probs), dim=-1)

            #     entropy = pad_input(
            #         hidden_states=entropy.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            #     )
            #     entropy = entropy.squeeze(-1)[:, -response_length - 1 : -1]
        else:
            if remove_images_for_logp: # build attention mask for logp mask掉 <|vision_start|><|image_pad|><|vision_end|>
                attention_mask = build_attention_mask(input_ids)
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs_to_pass,  # 使用条件化处理后的多模态输入
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]
            log_probs = self.log_probs_from_logits(logits, responses)

            # add here to calculate policy entropy
            # with torch.no_grad():
            #     ent_probs = F.softmax(logits, dim=-1)
            #     entropy = -torch.sum(ent_probs * torch.log(1e-8 + ent_probs), dim=-1)

        entropy = None

        return log_probs, entropy

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, _ = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    # log_probs, entropy = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs, entropy = self._forward_micro_batch(model_inputs, temperature=temperature)

                    # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    #     old_log_probs=old_log_probs,
                    #     log_probs=log_probs,
                    #     advantages=advantages,
                    #     eos_mask=response_mask,
                    #     cliprange=self.config.clip_ratio,
                    # )
                    pg_loss, pg_clipfrac, ppo_kl, rollout_util_rate, token_util_rate = core_algos.compute_policy_loss_v2(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=self.config.clip_ratio,
                    )
                    
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.kl_penalty(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        if not self.config.use_nll_loss:
                            pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        else:
                            pg_loss = pg_loss + kl_loss * self.config.kl_coef + nll_loss
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    # calculate entropy
                    # policy_entropy = torch.mean(VF.masked_mean(entropy, response_mask, dim=-1), dim=-1)

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/rollout_util_rate": rollout_util_rate,
                        "actor/token_util_rate": token_util_rate,
                        # "actor/policy_entropy": policy_entropy.item()
                    }
                    if self.config.use_nll_loss:
                        batch_metrics["actor/nll_loss"] = nll_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
