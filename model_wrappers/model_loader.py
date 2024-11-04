import argparse
import os
import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
# print(sys.path)

import gc
from PIL import Image
import torch

from transformers import ChameleonProcessor
from transformers import ChameleonForConditionalGeneration

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler
from scheduler.jacobi_iteration_anhole import renew_pipeline_sampler as renew_pipeline_sampler_anhole
from scheduler.jacobi_iteration_emu3 import renew_solver as renew_solver_emu3


def load_lumina_mgpt(
    cache_dir = "./ckpts",
    model_name = "Alpha-VLLM/Lumina-mGPT-7B-768",
    target_size = 768,
    seed = 1,
    max_num_new_tokens = 16,
    multi_token_init_scheme = 'random',
    guidance_scale = 7.0,
    device = "cpu",
    **kwargs,
):
    model_path = model_name

    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        precision="bf16",
        target_size=target_size,
        cache_dir=cache_dir,
        device = device,
    )
    

    print(inference_solver.__class__)
    inference_solver = renew_pipeline_sampler(
        inference_solver,
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 16)**2 + target_size // 16 - 10,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seed,
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg = True,
        **kwargs,
    )

    return inference_solver

def load_anole(
    cache_dir = "./ckpts",
    model_name = "leloy/Anole-7b-v0.1-hf",
    target_size = 512,
    seed = 1,
    max_num_new_tokens = 16,
    multi_token_init_scheme = 'random',
    guidance_scale = 7.0,
    device = "cpu",
    dtype = torch.bfloat16,
    image_top_k = 2000,
    text_top_k = 10,
    prefix_token_sampler_scheme = 'speculative_jacobi',
    **kwargs,
):

    processor = ChameleonProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    model = renew_pipeline_sampler_anhole(
        model,
        processor,
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 16)**2 + target_size // 16 - 10,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seed,
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        **kwargs,
    )

    inference_solver = dict(
        processor=processor,
        model=model,
    )
    
    return inference_solver

def load_emu3(
    cache_dir = "./ckpts",
    model_name = "BAAI/Emu3-Gen",
    target_size = 720,
    seed = 1,
    max_num_new_tokens = 16,
    multi_token_init_scheme = 'random',
    guidance_scale = 7.0,
    device = "cpu",
    dtype = torch.bfloat16,
    image_top_k = 2048,
    text_top_k = 10,
    prefix_token_sampler_scheme = 'speculative_jacobi',
    **kwargs,
):
    from emu3.mllm.processing_emu3 import Emu3Processor

    EMU_HUB = model_name
    VQ_HUB = "BAAI/Emu3-VisionTokenizer"

    model_name = EMU_HUB.split("/")[-1]
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation="sdpa", # "sdpa" # "flash_attention_2"
        trust_remote_code=True,
        cache_dir = cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, cache_dir=cache_dir,)
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, cache_dir=cache_dir,)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=device, trust_remote_code=True, cache_dir=cache_dir,).eval()

    image_tokenizer = image_tokenizer.to(dtype)
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    classifier_free_guidance = guidance_scale

    kwargs = dict(
        mode='G',
        ratio="1:1",
        image_area=model.config.image_area,
        return_tensors="pt",
    )
    GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=image_top_k,
    )

    h, w = target_size // 8, target_size // 8

    constrained_fn = processor.build_prefix_constrained_fn(h, w)

    model, logits_processor = renew_solver_emu3(
        model, processor, 
        h = h, w = w,
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = h * (w+1) - 1, 
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seed,
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        **kwargs,
    )

    inference_solver = dict(
        processor=processor,
        model=model,
        GENERATION_CONFIG=GENERATION_CONFIG,
        logits_processor=logits_processor,
    )
    
    return inference_solver

def load_llamagen(
    cache_dir = "./ckpts",
    model_name = "llamagen", 
    target_size = 512,
    seed = 1,
    max_num_new_tokens = 16, 
    multi_token_init_scheme = 'random',
    guidance_scale = 7.5,
    device = "cpu",
    dtype = torch.bfloat16,
    image_top_k = 1000,
    text_top_k = 10,
    prefix_token_sampler_scheme = 'speculative_jacobi',
    vq_params=dict(
        vq_model="VQ-16",
        codebook_size=16384,
        codebook_embed_dim=8,
        vq_ckpt = "llamagen/vq_ds16_t2i.pt",
        downsample_size=16,
    ),
    backbone_params=dict(
        gpt_model = 'GPT-XL',
        cls_token_num = 120,
        gpt_type = 't2i',
        t5_path = 'llamagen/t5-ckpt',
        t5_model_type = 'flan-t5-xl',
        t5_feature_max_len = 120,
        no_left_padding = False,
    ),
    is_compile = False,
    image_top_p = 1.0,
    temperature = 1.0,
    **kwargs,
):
    from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
    from llamagen.language.t5 import T5Embedder
    from llamagen.llamagen import GPT_models
    from llamagen.llamagen_solver import LlamaGenSolver, renew_llamagen
    from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler

    vq_ckpt = vq_params['vq_ckpt']
    vq_ckpt = os.path.join(cache_dir, vq_ckpt)
    if target_size == 256:
        gpt_ckpt = "llamagen/t2i_XL_stage1_256.pt"
    else:
        gpt_ckpt = "llamagen/t2i_XL_stage2_512.pt"

    gpt_ckpt = os.path.join(cache_dir, gpt_ckpt)
    t5_path = backbone_params['t5_path']
    t5_path = os.path.join(cache_dir, t5_path)

    codebook_embed_dim = vq_params['codebook_embed_dim']
    # create and load model
    vq_model = VQ_models[ vq_params['vq_model'] ](
        codebook_size= vq_params['codebook_size'],
        codebook_embed_dim= codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load( vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = dtype
    latent_size = target_size // vq_params['downsample_size']
    gpt_model = GPT_models[ backbone_params['gpt_model'] ](
        block_size=latent_size ** 2,
        cls_token_num= backbone_params['cls_token_num'],
        model_type= backbone_params['gpt_type'],
    ).to(device=device, dtype=precision)

    print(gpt_model.__class__)

    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = latent_size**2 - max_num_new_tokens - 2, 
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seed,
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
        **kwargs,
    )

    gpt_model.__class__ = renew_llamagen(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)
    gpt_model.__class__ = renew_sampler(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)

    checkpoint = torch.load( gpt_ckpt , map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if is_compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    if not os.path.exists(t5_path):
        os.makedirs(t5_path)

    assert os.path.exists(t5_path), f"t5 model path {t5_path} does not exist"
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=t5_path, 
        dir_or_name= backbone_params['t5_model_type'],
        torch_dtype=precision,
        model_max_length= backbone_params['t5_feature_max_len'],
    )

    model = LlamaGenSolver(
        model = gpt_model,
        image_top_k=image_top_k,
        image_top_p=image_top_p,
    )

    inference_solver = dict(
        model=model,
        gpt_model=gpt_model,
        t5_model = t5_model,
        vq_model = vq_model,
        vq_params = vq_params,
        backbone_params = backbone_params,
        latent_size = latent_size,
        guidance_scale = guidance_scale,
        temperature = temperature,
        image_top_k=image_top_k,
        image_top_p=image_top_p,
    )
    
    return inference_solver

def load_pretrained_model(
    model_name = "Alpha-VLLM/Lumina-mGPT-7B-768", **kwargs,
):
    if ('lumina-mgpt' in model_name.lower()):
        return load_lumina_mgpt(model_name=model_name, **kwargs)
    elif ('anole' in model_name.lower()):
        return load_anole(model_name=model_name, **kwargs)
    elif ('llamagen' in model_name.lower()):
        return load_llamagen(model_name=model_name, **kwargs)
    elif ('emu3' in model_name.lower()):
        return load_emu3(model_name=model_name, **kwargs)
    else:
        raise NotImplementedError


def get_lumina_mgpt_forward_func(
    inference_solver, 
    guidance_scale=7.0, 
    image_top_k=2000,
    max_gen_len=8192,
    temperature=1.0,
    target_size=768,
    **kwargs,
):

    def sample_fn(prompts):
        prompts = f"Generate an image of {target_size}x{target_size} according to the following prompt:\n" + prompts

        generated = inference_solver.generate(
            images=[],
            qas=[[prompts, None]],
            max_gen_len=max_gen_len,
            temperature=temperature,
            logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
        )
        a1, new_image = generated[0], generated[1][0]

        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        return result_image
    
    return sample_fn

def get_anole_forward_func(
    inference_solver, 
    **kwargs,
):
    processor = inference_solver['processor']
    model = inference_solver['model']

    def sample_fn(prompts):

        # Prepare a prompt
        prompt = "Generate an image of " + prompts

        # Preprocess the prompt
        inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

        generate_ids = model.generate(
            **inputs,
            multimodal_generation_mode="image-only",
            max_new_tokens=1026,
            do_sample=True,
        )

        # Only keep the tokens from the response
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

        # Decode the generated image tokens
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        images = processor.postprocess_pixel_values(pixel_values)

        # Save the image
        images = images[0].permute(1, 2, 0).cpu().numpy()
        result_image = Image.fromarray((images ).astype("uint8"))

        return result_image

    return sample_fn


def get_emu3_forward_func(
    inference_solver, 
    not_decoded_imgs=False,
    **kwargs,
):
    processor = inference_solver['processor']
    model = inference_solver['model']
    GENERATION_CONFIG = inference_solver['GENERATION_CONFIG']
    logits_processor = inference_solver['logits_processor']

    def sample_fn(prompts):

        POSITIVE_PROMPT = " masterpiece, film grained, best quality."
        NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        prompt = prompts
        prompt += POSITIVE_PROMPT

        pos_inputs = processor(text=prompt, **kwargs)
        neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

        device = model.device

        pos_input_ids = pos_inputs.input_ids
        neg_input_ids = neg_inputs.input_ids

        if not isinstance(pos_input_ids, torch.Tensor):
            pos_input_ids = torch.tensor(pos_input_ids).to(device)
            neg_input_ids = torch.tensor(neg_input_ids).to(device)
        else:
            pos_input_ids = pos_input_ids.to(device)
            neg_input_ids = neg_input_ids.to(device)

        model_inputs = model.prepare_batch_cfg_model_inputs(
            pos_input_ids, 
            neg_input_ids=neg_input_ids, 
            attention_mask=None,
        )
        pos_input_ids = model_inputs['pos_input_ids']
        attention_mask = model_inputs['attention_mask']


        outputs = model.generate(
            pos_input_ids,
            GENERATION_CONFIG,
            logits_processor=logits_processor, 
            attention_mask=attention_mask,
            neg_input_ids=neg_input_ids,
        )
        outputs = outputs[0]

        if not_decoded_imgs:
            result = outputs
        else:

            with torch.no_grad():
                mm_list = processor.decode(outputs)

            result_images = []
            for idx, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue

                result_images.append(im)
            
            result = result_images[-1]

        return result

    return sample_fn

def get_llamagen_forward_func(
    inference_solver, 
    **kwargs,
):
    from llamagen.llamagen_solver import generate as llamagen_original_generate
    model = inference_solver['model']
    gpt_model = inference_solver['gpt_model']
    t5_model = inference_solver['t5_model']
    vq_model = inference_solver['vq_model']
    latent_size = inference_solver['latent_size']
    vq_params = inference_solver['vq_params']
    backbone_params = inference_solver['backbone_params']
    guidance_scale = inference_solver['guidance_scale']
    temperature = inference_solver['temperature']
    image_top_k = inference_solver['image_top_k']
    image_top_p = inference_solver['image_top_p']

    codebook_embed_dim = vq_params['codebook_embed_dim']
    no_left_padding = backbone_params['no_left_padding']

    def sample_fn(prompts):

        prompts = [
            prompts, #"A blue Porsche 356 parked in front of a yellow brick wall.",
        ]

        caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

        if not no_left_padding:
            print(f"processing left-padding...")    
            # a naive way to implement left-padding
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                print(f'  prompt {idx} token len: {valid_num}')
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), codebook_embed_dim, latent_size, latent_size]

        index_sample = llamagen_original_generate(
            gpt_model,
            c_indices, latent_size ** 2, 
            c_emb_masks, 
            cfg_scale= guidance_scale,
            temperature= temperature, top_k= image_top_k,
            top_p= image_top_p, sample_logits=True, 
        )
        samples = vq_model.decode_code(index_sample, qzshape)
        
        images = samples
        images = images.clamp(min=-1, max=1)
        images = (images - images.min()) / (images.max() - images.min()) * 255
        images = images[0].permute(1, 2, 0).cpu().numpy()
        result_image = Image.fromarray((images ).astype("uint8"))

        return result_image

    return sample_fn

def get_forward_func(model_name, model, **kwargs):
    if ('lumina-mgpt' in model_name.lower()):
        return get_lumina_mgpt_forward_func(model, **kwargs)
    elif ('anole' in model_name.lower()):
        return get_anole_forward_func(model, **kwargs)
    elif ('llamagen' in model_name.lower()):
        return get_llamagen_forward_func(model, **kwargs)
    elif ('emu3' in model_name.lower()):
        return get_emu3_forward_func(model, **kwargs)
    else:
        raise NotImplementedError