import os
import math
from argparse import ArgumentParser
import time
import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

from typing import List, Optional, Union, Dict
from copy import copy

from utils import set_logger

class PromptWrapper:
	def __init__(
		self,
		eval_data: DataLoader,
		gpu_id,
		node_id,
		model_name = "Alpha-VLLM/Lumina-mGPT-7B-768",
		output_dir = "./workdir",
		seed = None,
	) -> None:

		self.gpu_id = gpu_id
		self.node_id = node_id
		self.device = torch.device(f"cuda:{self.gpu_id}")
		print(f"GPU {self.gpu_id} is initialized")
		self.eval_data = eval_data

		self.seed = seed
		# self.max_num_new_tokens = max_num_new_tokens
		self.model_name = model_name.split("/")[-1]
		
		self.output_dir = output_dir
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
	
	def run(self, sample_fn):
		for i, data_item in enumerate(tqdm(
			self.eval_data, desc=f"Generating captions on GPU {self.gpu_id}, Node {self.node_id}"
		)):
			prompt, prompt_idx = data_item

			prompt = prompt[0]
			prompt_idx = prompt_idx[0].item()

			output_file_name = str(prompt_idx) + ".png"
			output_file_path = self.output_dir + "/" + output_file_name
			if not os.path.exists(output_file_path):
				result_image = sample_fn(prompt)
				if isinstance(result_image, torch.Tensor):
					output_file_path = output_file_path.replace(".png", ".pt")
					torch.save(result_image, output_file_path)
				elif isinstance(result_image, Image.Image):
					result_image.save(output_file_path)
				else:
					raise ValueError(f"Invalid image type: {type(result_image)}")


from .dataset_templates import create_dataset
from model_wrappers.model_loader import load_pretrained_model, get_forward_func
def run_caption_gen( 
	gpu_id,
	node_id,
	gpu_ids,
	node_ids,
	dataset_params = dict(
		name = 'parti',
		annFile = './data/PartiPrompts.tsv',
	),
	seed = None,
	model_name = "Alpha-VLLM/Lumina-mGPT-7B-768",
	output_dir = "./workdir",
	**kwargs,
):
	dataset = create_dataset(
        gpu_id=gpu_id,
        gpu_ids=gpu_ids,
        node_id=node_id,
        node_ids=node_ids,
		output_dir=output_dir,
		**dataset_params,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		pin_memory=True,
		num_workers=12,
	)
	device = torch.device(f"cuda:{gpu_id}")
	print(f"device {device}, GPU {gpu_id} is initialized, running on Node {node_id}.")
	
	model = load_pretrained_model(
		model_name, 
		device = device,
		seed = seed,
		**kwargs,
	)

	forward_func = get_forward_func(
		model_name,
		model,
		**kwargs,
	)

	prompt_gen = PromptWrapper(
		eval_data=dataloader,
		gpu_id=gpu_id,
		node_id=node_id,
		seed = seed,
		model_name = model_name,
		output_dir = output_dir,
	)
	set_logger(log_level='info', fname=os.path.join(output_dir, 'gen_img_output.log'))
	with torch.no_grad():
		prompt_gen.run(forward_func)

def _run_on_gpu(
		gpu_id, 
		gpu_ids, 
		node_id,
		node_ids,
		kwargs):
	"""
	Function that calls run caption gen with the specified arguments.
	"""
	# Set the GPU ID for the process if needed (optional)
	# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	run_caption_gen(
		gpu_id=gpu_id, 
		node_id=node_id,
		gpu_ids=gpu_ids,
		node_ids=node_ids,
		**kwargs)


def _run_on_multiple_gpus(
		gpu_ids, 
		node_ids,
		node_id,
		**kwargs):
	"""
	Launches run caption gen on multiple GPUs without using multiprocessing.Pool,
	ensuring subprocesses are not daemonic and can have their CUDA context.
	
	Args:
	- num_gpus (int): Number of GPUs to use.
	- **kwargs: Arguments for the run caption gen function, excluding gpu_id.
	"""
	to_iterate = gpu_ids 


	processes = []
	for gpu_id in to_iterate:
		# Prepare the arguments for each GPU
		p = multiprocessing.Process(target=_run_on_gpu, 
							  		args=(gpu_id, gpu_ids, node_id, node_ids, kwargs), 
									daemon=False)
		p.start()
		processes.append(p)
	
	for p in processes:
		p.join()  # Wait for all processes to complete
