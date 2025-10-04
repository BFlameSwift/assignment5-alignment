from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from tests.personal import get_model_path


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    
    # return 
    # model = AutoModelForCausalLM.from_pretrained(get_model_path(),torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    # tokenizer = AutoTokenizer.from_pretrained(get_model_path())
    
    # prompt_ids = tokenizer.batch_encode_plus(prompt_strs)
    # output_ids = tokenizer.batch_encode_plus(output_strs)
    prompt_ids = [tokenizer.encode(id,add_special_tokens=False) for id in prompt_strs]
    output_ids = [tokenizer.encode(id,add_special_tokens=False) for id in output_strs]
    # breakpoint()
    # Concatenate prompt and output ids for each sample
    combined_ids = [torch.cat([torch.tensor(p), torch.tensor(o)]) for p, o in zip(prompt_ids, output_ids)]
    
    
    prompt_lens,output_lens = [len(ids) for ids in prompt_ids],[len(ids) for ids in output_ids]
    
    max_len = max([t.size(0) for t in combined_ids])
    b = len(prompt_lens)
    rows = torch.arange(b)
    
    # pids = torch.zeros(b, max_len)
    # oids = torch.zeros(b, max_len)
    # response_mask = torch.zeros(b,max_len,dtype=torch.bool)
    
    # for i,(prompt,output) in enumerate(zip(prompt_ids,output_ids)):
    #     plen,olen = len(prompt),len(output)
    #     response_mask[i][:plen] = True
    # response_mask[rows,:prompt_lens] = True
    # input_ids = torch.zeros(len(combined_ids), max_len, dtype=torch.long)
    # for i, ids in enumerate(combined_ids):
    #     input_ids[i, :len(ids)] = ids
    # breakpoint()
    
    response_mask = torch.zeros(b,max_len,dtype=torch.bool)
    
    input_ids = torch.ones(len(combined_ids), max_len, dtype=torch.long) 
    # breakpoint()
    input_ids *=  tokenizer.pad_token_id
    
    # labels = torch.ones(len(combined_ids), max_len, dtype=torch.long) * tokenizer.pad_token_id
    # response_mask[rows][:torch.tensor(prompt_lens)] = True
    for i, (prompt_len,output_len) in enumerate(zip(prompt_lens,output_lens)):
        # breakpoint()
        # 为什么是-1呢
        response_mask[i, prompt_len-1:prompt_len+output_len-1] = True
        
    # response_mask = torch.arange(max_len)[None, :] < prompt_lens_tensor[:, None]
    for i, ids in enumerate(combined_ids):
        input_ids[i][:ids.size(0)] = ids
    
    labels = input_ids[:,1:]
    input_ids = input_ids[:,:-1]
    response_mask = response_mask[:,:-1]
    
    # prompt_lens,output_lens = torch.
    
    # print()
    # breakpoint()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask" : response_mask
    }
    
    
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # raise NotImplementedError
    # torch.logsumexp()
    import torch.nn.functional as F

    # return -torch.logsumexp(logits ,dim=-1)
    # breakpoint()
    # return -torch.logsumexp(probs  * torch.log(probs) ,dim=-1)
    # Methods 1
    probs = F.softmax(logits, dim=-1)  
    # return -torch.sum(probs  * torch.log(probs) ,dim=-1)
    
    # Methods2
    logprob = logits - torch.logsumexp( logits,dim=-1,keepdim=True)
    # 相同
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # breakpoint()
    # breakpoint()
    return -torch.sum(logprob * torch.exp(logprob),dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    batch_size, sequence_length = input_ids.shape
    
    outputs = model(input_ids)
    
    rows = torch.arange(batch_size).unsqueeze(1)  # Shape: (batch_size, 1)
    cols = torch.arange(sequence_length).unsqueeze(0)  # Shape: (1, sequence_length)
    
    # Get log probabilities for the actual labels
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    selected_log_probs = log_probs[rows, cols, labels]  # Shape: (batch_size, sequence_length)

    
    # breakpoint()
    ret = {}
    ret['log_probs'] = selected_log_probs
    
    if return_token_entropy:
        ret['token_entropy'] = compute_entropy(outputs.logits )
        # pass
        # breakpoint()
    return ret
    
    
    
    