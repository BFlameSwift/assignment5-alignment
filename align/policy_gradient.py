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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    
    return - raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    # raise NotImplementedError
    
    native_pg_loss = compute_naive_policy_gradient_loss(advantages,policy_log_probs)
    
    
    log_probs_ratio = policy_log_probs - old_log_probs
    probs_ratio = torch.exp(log_probs_ratio)
    # breakpoint()
    # clip_ratio = min(probs_ratio, 1+cliprange) if probs_ratio.item() > 0 else max(probs_ratio, 1- cliprange)
    
    below_clip = probs_ratio <=  (1- cliprange)
    up_clip =  probs_ratio >=  (1 + cliprange)
    
    one = torch.ones_like(below_clip,dtype=torch.long)
    
    clip_ratio = one * (1- cliprange) * below_clip + one * (1+ cliprange) * up_clip + ~(below_clip+up_clip) * probs_ratio
    # clip_ratio = torch.clamp(probs_ratio, 1 - cliprange, 1 + cliprange)
    


    # clip_ratio = torch.where(below_clip, 1 - cliprange,
    #                 torch.where(up_clip, 1 + cliprange, probs_ratio))
    # breakpoint()
    
    metadata = {
        "clip_fraction": (torch.abs(probs_ratio - clip_ratio) > 0).float().mean(),
        "mean_ratio": probs_ratio.mean(),
        "std_ratio": probs_ratio.std(),
        "max_ratio": probs_ratio.max(),
        "upper_cliped_ratio_rate": up_clip.float().mean(),
        "lower_cliped_ratio_rate": below_clip.float().mean(),
        "min_ratio": probs_ratio.min(),
        "mean_advantage": advantages.mean(),
        "std_advantage": advantages.std(),
        "loss_type":"grpo_clip"
        
    }
    # if cliprange < 0.5:
    #     breakpoint()
    
    return - torch.min(clip_ratio* advantages,probs_ratio*advantages) , metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    # raise NotImplementedError
    
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        # breakpoint()
        # loss = loss.mean()
        return loss,{"loss_type": "reinforce_with_baseline",'loss':loss.mean()}
    elif loss_type == "no_baseline":
        
        loss =  compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs)
        return  loss, {"loss_type": "no_baseline",'loss':loss.mean()}
    else:
        raise Exception("not a type")
    
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # breakpoint()
    # return (tensor * mask).mean(dim=dim)
    mat = tensor * mask
    
    return mat.sum(dim=dim) / mask.sum(dim=dim)