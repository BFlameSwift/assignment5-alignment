# %%
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable, List

from drgrpo_grader import r1_zero_reward_fn
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# %%
def preprocess_MATH_dataset(filepath, prompt_key="problem", answer_key="solution"):
	prompts = []
	answers = []
	
	with xopen(filepath, 'r') as f:
		for line in f:
			data = json.loads(line.strip())
			prompts.append(data[prompt_key])
			answers.append(data[answer_key])
	
	return prompts, answers
def get_data_hour_str() -> str:
	return datetime.now().strftime("%d_%H%M")

# %%
MATH_validation_path = '../data/MATH/validation.jsonl'


# %%
r1_zero_prompt ="""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def preprocess_prompts(prompts,default_prompt=r1_zero_prompt):
	ret = [default_prompt.format(question=p) for p in prompts]
	return ret

def evaluate_vllm( vllm_model: LLM, reward_fn: Callable[[str, str], dict[str, float]], prompts: List[str], eval_sampling_params: SamplingParams,output_path=None,solutions=None) -> None: 
    # """ 
    # Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk. 
    # """
	outputs = vllm_model.generate(prompts, eval_sampling_params)

	if isinstance(output_path,str):
		output_path = Path(output_path)
	if output_path is None:
		output_path = Path("../data/output") / (get_data_hour_str()+".jsonl")

	reward_results = []
	format_reward,answer_reward,reward = 0.0,0.0,0.0

	with open(output_path,'w') as f:
		for idx,output in enumerate(outputs): 
			prompt = output.prompt 
			generated_text = output.outputs[0].text 
			json_line = {"prompt":prompt,"predict":generated_text}
			f.write(json.dumps(json_line) + '\n')

			r = reward_fn(generated_text,solutions[idx])
			format_reward += r['format_reward']
			answer_reward += r['answer_reward']
			reward += r['reward']

			reward_results.append(r)
	# breakpoint()
	format_reward /= len(reward_results)
	answer_reward/= len(reward_results)
	reward /= len(reward_results)
	print(f"Format Reward: {format_reward:.4f}")
	print(f"Answer Reward: {answer_reward:.4f}")
	print(f"Total Reward: {reward:.4f}")
	with open(str(output_path).replace(".jsonl","_reward.json"),'w') as f:
		json.dumps(f,reward_results,indent=4,ensure_ascii=False)
    

# %%
llm = LLM("../models/Qwen2.5-Math-1.5B")
prompts,solutions = preprocess_MATH_dataset(MATH_validation_path)

sampling_params = SamplingParams( temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"] )
evaluate_vllm(llm,r1_zero_reward_fn,preprocess_prompts(prompts),sampling_params,solutions=solutions,output_path="../data/output/math_qwen2.5_1.5b_"+get_data_hour_str()+".jsonl")


