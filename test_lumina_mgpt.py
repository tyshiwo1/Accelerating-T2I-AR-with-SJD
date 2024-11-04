import argparse
import os
import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
print(sys.path)

import gc

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from PIL import Image
import torch
import time

import random
import numpy as np

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# CUDA_VISIBLE_DEVICES=0 python test_lumina_mgpt.py

cache_dir = "./ckpts/"

# model_path = "Alpha-VLLM/Lumina-mGPT-7B-512"
# target_size = 512

model_path = "Alpha-VLLM/Lumina-mGPT-7B-768"
target_size = 768
target_size_h, target_size_w = 768, 768

# model_path = "Alpha-VLLM/Lumina-mGPT-7B-1024"
# target_size = 1024

# model_path = "Alpha-VLLM/Lumina-mGPT-34B-512"
# target_size = 512

device = "cuda:0"

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path=model_path,
    precision="bf16",
    target_size=target_size,
    cache_dir=cache_dir,
    device = device,
)

seeds = [None, ] #[_ for _ in range(124, 200) ]
max_num_new_tokens = 16
multi_token_init_scheme = 'random' # 'repeat_horizon'
image_top_k = 2000 
text_top_k = 10
guidance_scale = 3.0
prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

q_image_content_conditions = [
    "A typical zebra standing majestically in the grassland, with its upper body fully visible and unobscured by grass. The zebra features two distinct ears and a clear, intricate striped pattern. This is a professionally captured, high-quality 8K photograph, showcasing impeccable clarity, sharp focus, and exquisite details in every texture.",
    "A strikingly beautiful girl with deep red eyes, short white hair, and a sly smile, dressed in elegant purple attire with a hood. The image features perfect eye-symmetry and facial-symmetry, captured in stunning 8K resolution for unparalleled high quality and realism.",
    "One lynx in the forest is illuminated by a gloomy strong light, the most Professional high-quality 8K photograph",
    "Atlantis, the most Fantasy high-quality photos",
    "a giant golden flying saucer firing lasers from the bottom, scorching the ground, the most Fantasy high-quality photos",
    "a cool man with a beautiful face wearing a yellow suit stands in the Mountain, the most Professional high-quality 8K photograph",
    "gel wax candle above the sea at night, analog photo,photoart_style,realistic,film grain,4k,volumetric natural light, epic atmosphere,perfect composition, highly detail, enchanted,deep rich vivid colors, perfect symmetry,  great composition, complimentary colors, beautiful elegant stylish, intricate detail",
    "IA huge black bear who converted to Buddhism, looking strong but furry and a little fierce, wearing a red cassock, before a blazing fire, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k.",
    "Portrait of a fairy wearing a pink top and apricot flowers in her bun, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k.",
    "An old elephant wearing Chinese armor, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k.",
    "A huge moon cake on a clean table, great composition, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k.",
    "Miss Mexico portrait of the most beautiful mexican woman, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k",
    "portrait of the most beautiful aisan woman, Wearing a dress and headdress decorated with peacock feathers, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k",
    "A golden-winged large fabulous bird with sunset, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k.",
    "A giant golden-haired lion with an indigo face roars at the gate of heaven, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k",
    "Portrait of an ancient Chinese boy with big eyes, muscular body, head lowered arrogantly, can use fire magic, with red sky in the background, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k.",
    "The Imaginary Pure Land of Asia",
    "Professional photograph of Water Curtain Cave at flower-fruit mountain, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k.",
    "Portrait of a strong, muscular Asian hero with short red curly beard, bald head, wearing green and purple clothes, with yellow quicksand in the background, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k.",
    "2D logo of a pure white box in a pure black background",
    "The most fantastic fully-body portrait of a panda sits by the mirror-like shallow water with a layer of mist above the water surface at sunset. high-quality, 8K, facial-symmetry, Exquisite details",
    "A Corgi dog in 2D logo style, simple texture, clean background, facial- and eye-symmetry",
    "most beautiful anime artwork, a most cute anime girl, double exposure, iridescent nebula galaxy, black background, ethereal glow, bloom, hdr, high-quality, 8K",
    "Macro photography of a transparent water drop in the shape of a cat.",
    "A hawk-man with a red head",
    "A cool furry black monkey meditates on the clean wet ground, in the dusk, the golden sunset is shining on the ground on one side and the other side, high-quality, 8K, facial-symmetry",
    "A masterpiece of oil painting about the starry sky",
    "An oil painting of a lady",
    "a cat on a mat"
]

template_condition_sentences = [
    f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n",
] * len(q_image_content_conditions)

from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
print(inference_solver.__class__)
inference_solver = renew_pipeline_sampler(
    inference_solver,
    jacobi_loop_interval_l = 3,
    jacobi_loop_interval_r = (target_size // 16)**2 + target_size // 16 - 10,
    max_num_new_tokens = max_num_new_tokens,
    guidance_scale = guidance_scale,
    seed = seeds[0],
    multi_token_init_scheme = multi_token_init_scheme,
    do_cfg=  True,
    image_top_k=image_top_k, 
    text_top_k=text_top_k,
    prefix_token_sampler_scheme = prefix_token_sampler_scheme,
)

for seed in seeds:
    inference_solver.model.seed = seed
    for i, q_image_content_condition in enumerate(q_image_content_conditions):
        q1 = template_condition_sentences[i] + q_image_content_condition

        output_file_name = model_path.split("/")[-1] + "-" + q_image_content_condition[:30] + '-' + str(max_num_new_tokens) + '-init-' + multi_token_init_scheme[:6] + '-seed' + str(seed) + '-img_topk' + str(image_top_k) + ".png"

        time_start = time.time()
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t1.record()

        generated = inference_solver.generate(
            images=[],
            qas=[[q1, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
        )
        t2.record()
        torch.cuda.synchronize()

        t = t1.elapsed_time(t2) / 1000
        time_end = time.time()
        print("Time elapsed: ", t, time_end - time_start)

        a1, new_image = generated[0], generated[1][0]

        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        result_image.save("./workdir/" + output_file_name)
        print(a1, 'saved', output_file_name) # <|image|>


del inference_solver
gc.collect()