import os
import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
print(sys.path)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

import time
import argparse

from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
from llamagen.language.t5 import T5Embedder
from llamagen.llamagen import GPT_models
from llamagen.llamagen_solver import LlamaGenSolver, renew_llamagen, generate
from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler

from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_jacobi_param_dict():
    target_size = 512

    seeds = [None, ]
    max_num_new_tokens =16 
    multi_token_init_scheme = 'repeat_horizon'
    image_top_k = 1000
    text_top_k = 10
    guidance_scale = 7.5
    prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 16)**2 - max_num_new_tokens - 2, 
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
    )
    return jacobi_param_dict

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(gpt_model.__class__)

    jacobi_param_dict = get_jacobi_param_dict()
    image_top_k = jacobi_param_dict['image_top_k']

    gpt_model.__class__ = renew_llamagen(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)
    gpt_model.__class__ = renew_sampler(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
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

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    if not os.path.exists(args.t5_path):
        os.makedirs(args.t5_path)

    assert os.path.exists(args.t5_path), f"t5 model path {args.t5_path} does not exist"
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    prompts = [
        "a big purple bus parked in a parking spot",
        # "A blue Porsche 356 parked in front of a yellow brick wall.",
        # "a photo of a teapot in a garden. teapot texture: transparent; color: red; shape: pumpkin.",
    ]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    if not args.no_left_padding:
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

    solver = LlamaGenSolver(
        model = gpt_model,
        image_top_k=image_top_k,
        image_top_p=args.top_p,
    )

    print()
    print(f"start sampling...")
    print()

    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()
    index_sample = solver.generate(
        c_indices, latent_size ** 2, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=image_top_k,
        top_p=args.top_p, sample_logits=True, 
    )
    # index_sample = generate(
    #     gpt_model,
    #     c_indices, latent_size ** 2, 
    #     c_emb_masks, 
    #     cfg_scale=args.cfg_scale,
    #     temperature=args.temperature, top_k=image_top_k,
    #     top_p=args.top_p, sample_logits=True, 
    # )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    print(samples)
    images = samples
    images = images.clamp(min=-1, max=1)
    images = (images - images.min()) / (images.max() - images.min()) * 255
    images = images[0].permute(1, 2, 0).cpu().numpy()
    result_image = Image.fromarray((images ).astype("uint8"))
    result_image.save(f"sample_{args.gpt_type}.png")
    print(f"image is saved to sample_{args.gpt_type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5_path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5_model_type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5_feature_max_len", type=int, default=120)
    parser.add_argument("--t5_feature_dim", type=int, default=2048)
    parser.add_argument("--no_left_padding", action='store_true', default=False)
    parser.add_argument("--gpt_model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt_ckpt", type=str, default=None)
    parser.add_argument("--gpt_type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls_token_num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq_model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample_size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)

    # CUDA_LAUNCH_BLOCKING=1 python3 tests/test_llamagen.py --vq_ckpt ./ckpts/llamagen/vq_ds16_t2i.pt --gpt_ckpt ./ckpts/llamagen/t2i_XL_stage2_512.pt --gpt_model GPT-XL --image_size 512 --t5_path ./ckpts/llamagen/t5-ckpt --top_p 1.0 --cfg_scale 3.5