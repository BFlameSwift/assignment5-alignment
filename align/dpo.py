from __future__ import annotations

import os
from pickle import FALSE
from typing import Any, Callable, Literal

import torch
from einops import einsum, rearrange
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from tests.personal import get_model_path

propmt_template = \
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""   


def compute_per_instance_dpo_loss_other(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    input_pos = propmt_template.format(prompt=prompt, response=response_chosen)+(tokenizer.eos_token or "")
    input_neg = propmt_template.format(prompt=prompt, response=response_rejected)+(tokenizer.eos_token or "")

    input_pos_ids = tokenizer.encode(input_pos, return_tensors="pt")
    input_neg_ids = tokenizer.encode(input_neg, return_tensors="pt")

    def get_log_probs(model: torch.nn.Module, input_ids: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        if inference_mode:
            with torch.no_grad():
                logits = model(input_ids).logits
        else:
            logits = model(input_ids).logits
            
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:]
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return log_probs.sum(dim=-1)
    
    log_probs_pos = get_log_probs(lm, input_pos_ids, inference_mode=False)
    log_probs_neg = get_log_probs(lm, input_neg_ids, inference_mode=False)
    log_probs_pos_ref = get_log_probs(lm_ref, input_pos_ids, inference_mode=True)
    log_probs_neg_ref = get_log_probs(lm_ref, input_neg_ids, inference_mode=True)

    loss_pos = log_probs_pos - log_probs_pos_ref
    loss_neg = log_probs_neg - log_probs_neg_ref
    loss = -torch.nn.functional.logsigmoid(beta * (loss_pos - loss_neg))

    return loss

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # raise NotImplementedError
    from .sft import get_response_log_probs, tokenize_prompt_and_output

    # prompt = [prompt]
    # response_chosen = [response_chosen]
    # response_rejected = [response_rejected]
    
    chosen_input_ids,chosen_labels, chosen_mask = tokenize_prompt_and_output(prompt,response_chosen,tokenizer)["input_ids"],tokenize_prompt_and_output(prompt,response_chosen,tokenizer)["labels"] , tokenize_prompt_and_output(prompt,response_chosen,tokenizer)["response_mask"]
    rejected_input_ids,rejected_labels, rejected_mask = tokenize_prompt_and_output(prompt,response_rejected,tokenizer)["input_ids"],tokenize_prompt_and_output(prompt,response_rejected,tokenizer)["labels"] , tokenize_prompt_and_output(prompt,response_rejected,tokenizer)["response_mask"]
    
    
    # breakpoint()
    
    lm_chosed_log_probs = get_response_log_probs(lm,chosen_input_ids,chosen_labels,False)["log_probs"]
    lm_ref_chosed_log_probs = get_response_log_probs(lm_ref,chosen_input_ids,chosen_labels,False)["log_probs"]
    
    lm_rejected_log_probs = get_response_log_probs(lm, rejected_input_ids, rejected_labels,False)["log_probs"]
    lm_ref_rejected_log_probs = get_response_log_probs(lm_ref, rejected_input_ids, rejected_labels,False)["log_probs"]
    
    # breakpoint()
    
    # lm_chosed_log_probs *= chosen_mask
    # lm_ref_chosed_log_probs *= chosen_mask
    # lm_rejected_log_probs *= rejected_mask
    # lm_ref_rejected_log_probs *= rejected_mask
    
    
    chosen_ratio = lm_chosed_log_probs - lm_ref_chosed_log_probs
    rejected_ratio = lm_rejected_log_probs - lm_ref_rejected_log_probs
    # breakpoint()
    input_pos = propmt_template.format(prompt=prompt, response=response_chosen)+(tokenizer.eos_token or "")
    input_neg = propmt_template.format(prompt=prompt, response=response_rejected)+(tokenizer.eos_token or "")

    input_pos_ids = tokenizer.encode(input_pos, return_tensors="pt")
    input_neg_ids = tokenizer.encode(input_neg, return_tensors="pt")

    def get_log_probs(model: torch.nn.Module, input_ids: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        if inference_mode:
            with torch.no_grad():
                logits = model(input_ids).logits
        else:
            logits = model(input_ids).logits
            
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:]
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return log_probs.sum(dim=-1)
    
    log_probs_pos = get_log_probs(lm, input_pos_ids, inference_mode=False)
    log_probs_neg = get_log_probs(lm, input_neg_ids, inference_mode=False)
    log_probs_pos_ref = get_log_probs(lm_ref, input_pos_ids, inference_mode=True)
    log_probs_neg_ref = get_log_probs(lm_ref, input_neg_ids, inference_mode=True)
    
    
    breakpoint()
    lm_chosed_log_probs = log_probs_pos
    lm_ref_chosed_log_probs = log_probs_pos_ref
    lm_rejected_log_probs = log_probs_neg
    lm_ref_rejected_log_probs = log_probs_neg_ref  
    
    chosen_ratio = lm_chosed_log_probs - lm_ref_chosed_log_probs
    rejected_ratio = lm_rejected_log_probs - lm_ref_rejected_log_probs


    dpo_loss  = -torch.log(torch.sigmoid(beta * (chosen_ratio - rejected_ratio)))
    # dpo_loss_mean = -torch.log(torch.sigmoid(beta * (chosen_ratio.mean() - rejected_ratio.mean())))
    import torch.nn.functional as F

    # dpo_loss_log_sigmoid = -F.logsigmoid(beta * (chosen_ratio.sum() - rejected_ratio.sum()))
    # breakpoint()
    return dpo_loss
    
    # Compute DPO loss
    # chosen_ratio = lm_chosed_log_probs - lm_ref_chosed_log_probs
    # rejected_ratio = lm_rejected_log_probs - lm_ref_rejected_log_probs
    
    # dpo_loss = -torch.log(torch.sigmoid(beta * (chosen_ratio - rejected_ratio)))
    
    # return dpo_loss
    # 