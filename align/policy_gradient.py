from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from einops import einsum, rearrange
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from tests.personal import get_model_path


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # breakpoint()
    
    # reward, format_reward, answer_reward = reward_fn(rollout_responses,repeated_ground_truths)["reward"], reward_fn(rollout_responses,repeated_ground_truths)["format_reward"], reward_fn(rollout_responses,repeated_ground_truths)["answer_reward"]
    rollout_batch_size = len(rollout_responses)
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert n_prompts_per_rollout_batch == rollout_batch_size /  group_size
    
    
    rewards,format_rewards,answer_rewards = torch.zeros(rollout_batch_size,dtype=torch.float32),torch.zeros(rollout_batch_size,dtype=torch.float32),torch.zeros(rollout_batch_size,dtype=torch.float32)
    for idx,(response,gt) in  enumerate(zip(rollout_responses,repeated_ground_truths)):
        reward, format_reward, answer_reward =  reward_fn(response, gt)["reward"], reward_fn(response, gt)["format_reward"], reward_fn(response, gt)["answer_reward"]
        rewards[idx] = reward
        format_rewards[idx] = format_reward
        answer_rewards[idx] = answer_reward
        
    rewards = rearrange(rewards, "(n g) -> n g",n=n_prompts_per_rollout_batch,g=group_size)
    format_rewards = rearrange(format_rewards, "(n g) -> n g",n=n_prompts_per_rollout_batch,g=group_size)
    answer_rewards = rearrange(answer_rewards, "(n g) -> n g",n=n_prompts_per_rollout_batch,g=group_size)
    
    # breakpoint()

    
    advs: Tensor = rewards - torch.mean(rewards,dim=-1,keepdim=True) 
    if normalize_by_std:
        advs /= (advantage_eps + torch.std(rewards,dim=-1,keepdim=True) )
        
    reward_mean = rewards.mean()
    format_rewards_mean = format_rewards.mean()
    answer_rewards_mean = answer_rewards.mean()
    advs_mean = advs.mean()
    
    metadata = {
        "reward_mean": reward_mean.item(),
        "format_reward_mean": format_rewards_mean.item(),
        "answer_reward_mean": answer_rewards_mean.item(),
        "advantage_mean": advs_mean.item(),
        "reward_std": rewards.std().item(),
        "advantage_std": advs.std().item()
    }
    
    rewards1d = rearrange(rewards, "n g -> (n g)",n=n_prompts_per_rollout_batch,g=group_size)
    advs1d = rearrange(advs, "n g -> (n g)",n=n_prompts_per_rollout_batch,g=group_size)
    # breakpoint()
    return advs1d, rewards1d,  metadata