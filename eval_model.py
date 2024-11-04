import ml_collections
import numpy as np
import os
import torch
from torch import multiprocessing as mp
import accelerate

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import tempfile
from torchvision.utils import save_image

import math
from argparse import ArgumentParser
import time
import multiprocessing

from dataset_tools.multi_gpu_infer_with_prompt import _run_on_multiple_gpus

from absl import logging
from utils import set_logger

if __name__ == "__main__":

	# set start method as 'spawn' to avoid CUDA re-initialization issues
	multiprocessing.set_start_method('spawn')

	parser = ArgumentParser()
	parser.add_argument("-v", "--verbose", action="store_true")


	parser.add_argument("--multiprocess", action="store_true")
	parser.add_argument("--gpu_ids", type=lambda x: [int(i) for i in x.split(",")], default=[0,1,2,3,])

	parser.add_argument(
		"--cache_dir",
		type=str,
		default="./ckpts",
	 	help="The directory to store the cache files."
	)

	parser.add_argument(
		"--node_id",
		type=int,
		default=0,
		help="Node ID for distributed inference."
	)

	parser.add_argument(
		"--node_ids",
		type=lambda x: [int(i) for i in x.split(",")],
		default=[0],
		help="Node IDs for distributed inference, separated by commas."
	)

	parser.add_argument(
		"--dataset_name",
		type=str,
		default="parti",
	)
	parser.add_argument(
		"--dataset_anno_file",
		type=str,
		default="./data/PartiPrompts.tsv",
	)

	parser.add_argument(
		"--model_name",
		type=str,
		default="Alpha-VLLM/Lumina-mGPT-7B-768",
	)

	parser.add_argument(
		"--max_num_new_tokens",
		type=int,
		default=16,
	)

	parser.add_argument(
		"--multi_token_init_scheme",
		type=str,
		default='sample_horizon', # sample_horizon sample_vertical # '2d_repeat' 'random' #'2d_extrapolation', #'repeat_last' # '1d_extrapolation' 
	)

	parser.add_argument(
		"--seed",
		type=int,
		default=1,
	)

	parser.add_argument(
		"--image_top_k",
		type=int,
		default=2000,
	)

	parser.add_argument(
		"--target_size",
		type=int,
		default=0,
	)

	parser.add_argument(
		"--prefix_token_sampler_scheme",
		type=str,
		default='speculative_jacobi',
	)

	parser.add_argument(
		"--guidance_scale",
		type=float,
		default=3.0,
	)

	args = parser.parse_args()

	start_time = time.time()

	max_num_new_tokens = args.max_num_new_tokens
	multi_token_init_scheme = args.multi_token_init_scheme
	seed = args.seed if args.seed >=0 else None
	model_name = args.model_name
	dataset_name = args.dataset_name
	guidance_scale = args.guidance_scale #3.0
	image_top_k = args.image_top_k
	prefix_token_sampler_scheme = args.prefix_token_sampler_scheme

	if args.target_size > 0:
		target_size = args.target_size
	else:
		potential_target_size = model_name.split("-")[-1]
		if potential_target_size.isdigit():
			target_size = int(potential_target_size)
		else:
			target_size = 512

	workdir = "./workdir_" + dataset_name + '-' + str(max_num_new_tokens) + '-seed' + str(seed) + '-' + multi_token_init_scheme + '-' + model_name.split("/")[-1] + '-' + str(target_size) + 'px' + '-cfg-' + str(guidance_scale)
	workdir = workdir + '-topk' + str(image_top_k)
	if prefix_token_sampler_scheme != 'speculative_jacobi':
		workdir = workdir + '-' + prefix_token_sampler_scheme

	if not os.path.exists(workdir):
		os.makedirs(workdir)

	set_logger(log_level='info', fname=os.path.join(workdir, 'gen_img_output.log'))
	

	logging.info(f"cache dir: {args.cache_dir}")
	logging.info(f"gpu_ids: {args.gpu_ids}")
	logging.info(f"node_ids: {args.node_ids}")
	logging.info(f"target_size: {target_size}")

	_run_on_multiple_gpus(
		gpu_ids=args.gpu_ids,
		node_ids=args.node_ids,
		node_id=args.node_id,
		\
		dataset_params = dict(
			name = args.dataset_name,
			annFile = args.dataset_anno_file,
		),
		model_name = args.model_name,
		\
		cache_dir = args.cache_dir,
		target_size = target_size,
		seed = seed,
		max_num_new_tokens = max_num_new_tokens,
		multi_token_init_scheme = multi_token_init_scheme,
		guidance_scale = guidance_scale,
		image_top_k=image_top_k,
		max_gen_len=8192,
		temperature=1.0,
		output_dir = workdir,
		prefix_token_sampler_scheme = prefix_token_sampler_scheme,
	)
	end_time = time.time()
	logging.info(f"Total Time taken: {end_time - start_time}")