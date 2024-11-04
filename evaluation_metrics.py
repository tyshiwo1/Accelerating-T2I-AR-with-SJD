from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers.models.attention_processor import Attention
import time

from argparse import ArgumentParser
import time
import multiprocessing

from absl import logging

from dataset_tools.dataset_templates import create_dataset
from utils import set_logger


def evaluate_template_matching(order_list, cal_cost, pipe):
    count = {0: 0}
    for pattern_type in order_list:
        count[pattern_type] = 0
    mask_count_total = 0
    total = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            for timestep, mask_list in module.mask.items():
                pattern_list = np.zeros(16)
                for i in range(16):
                    type = 0
                    for j in order_list:
                        if mask_list[j][i]:
                            type = j
                            pattern_list[i] = j
                            break
                    count[type] += 1
                module.mask[timestep] = pattern_list

    total_num = sum(count.values())
    cal = 0
    for k, v in count.items():
        cal += cal_cost[k] * v / total_num
    print("template matching info: ")
    print(count)
    print("total percentage reduction: ", round(1 - cal, 2))


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (299, 299))


def save_output_hook(m, i, o):
    m.saved_output = o


def test_latencies(pipe, n_steps, calib_x, bs, only_transformer=True, test_attention=True):
    latencies = {}
    for b in bs:
        pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        st = time.time()
        for i in range(3):
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        ed = time.time()
        t = (ed - st) / 3
        if only_transformer:

            handler = pipe.transformer.register_forward_hook(save_output_hook)
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=1)
            handler.remove()
            old_forward = pipe.transformer.forward
            pipe.transformer.forward = lambda *arg, **kwargs: pipe.transformer.saved_output
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other = (ed - st) / 3
            pipe.transformer.forward = old_forward
            del pipe.transformer.saved_output
            print(f"average time for other bs={b} inference: {t_other}")
            latencies[f"{b}_other"] = t_other
            latencies[f"{b}_transformer"] = t - t_other
        print(f"average time for bs={b} inference: {t}")
        latencies[f"{b}_all"] = t

        if test_attention:  # Test the latency of the attention modules
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.old_forward = module.forward
                    module.forward = lambda *arg, **kwargs: arg[0]
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other2 = (ed - st) / 3
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.forward = module.old_forward
            t_attn = t - t_other2
            print(f"average time for attn bs={b} inference: {t_attn}")
            latencies[f"{b}_attn"] = t_attn
    return latencies


def evaluate_quantitative_scores(
    pipe,
    real_image_path,
    n_images=50000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",
    guidance_scale=4,
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # Inception Score
    inception = InceptionScore().to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path):
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for i in range(0, n_images, batchsize):
        class_ids = np.random.randint(0, 1000, batchsize)
        output = pipe(
            class_labels=class_ids,
            generator=generator,
            output_type="np",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        fake_images = output.images
        # Inception Score
        torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
        torch_images = torch.nn.functional.interpolate(
            torch_images, size=(299, 299), mode="bilinear", align_corners=False
        ).to(device)
        inception.update(torch_images)

        for j, image in enumerate(fake_images):
            image = F.to_pil_image(image)
            image.save(f"{fake_image_path}/{i+j}.png")

    IS = inception.compute()
    results["IS"] = IS
    print(f"Inception Score: {IS}")

    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path],
        64,
        device,
        dims=2048,
        num_workers=8,
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results


def evaluate_quantitative_scores_text2img(
    pipe,
    real_image_path,
    mscoco_anno,
    n_images=5000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images", # reuse_generated=True,
    negative_prompt="",
    guidance_scale=4.5,
    name_format="pad_png",
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if real_image_path is not None:
        fid_value = calculate_fid_given_paths(
            [real_image_path, fake_image_path],
            1, #64,
            device,
            dims=2048,
            num_workers=0, #8,
        )
        results["FID"] = fid_value
        print(f"FID: {fid_value}")

    # Inception Score
    inception = InceptionScore().to(device)
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    # if os.path.exists(fake_image_path) and not reuse_generated:
    #     os.system(f"rm -rf {fake_image_path}")
    # os.makedirs(fake_image_path, exist_ok=True)

    img_type = name_format.split("_")[1]

    for index in range(0, n_images, batchsize):

        slice = mscoco_anno["annotations"][index : index + batchsize]
        print(f"Processing {index}th image")
        caption_list = [d["caption"] for d in slice]
        
        filename_list = []
        for d in slice:
            img_name = str(d["id"])
            if name_format.split("_")[0] == "pad":
                img_name = img_name.zfill(12)

            filename_list.append( img_name )

        torch_images = []
        for filename in filename_list:
            image_file = f"{fake_image_path}/{filename}.{img_type}"
            if os.path.exists(image_file):
                image = Image.open(image_file)
                image_np = np.array(image)
                torch_image = torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2)
                torch_images.append(torch_image)
            else:
                print(image_file)
        
        if len(torch_images) > 0:
            torch_images = torch.cat(torch_images, dim=0)
            print(torch_images.shape)
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list[: len(torch_images)])
        else:
            output = pipe(
                caption_list,
                generator=generator,
                output_type="np",
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
            )
            fake_images = output.images
            # Inception Score
            count = 0
            torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list)
            for j, image in enumerate(fake_images):
                # image = image.astype(np.uint8)
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{fake_image_path}/{filename_list[count]}.jpg")
                count += 1

    IS = inception.compute()
    CLIP = clip.compute()
    results["IS"] = IS
    results["CLIP"] = CLIP
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")
    
    return results


if __name__ == "__main__":

    # set start method as 'spawn' to avoid CUDA re-initialization issues
    multiprocessing.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument("--workdir", type=str, default="./workdir_parti-16",)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="parti_cocoformat", # coco
    )
    parser.add_argument(
        "--dataset_anno_file",
        type=str,
        default="./data/PartiPrompts.tsv", # 'data/coco/annotations/captions_val2017.json'
    )

    args = parser.parse_args()
    workdir = args.workdir
    annFile = args.dataset_anno_file
    dataset_name = args.dataset_name

    gpu_id = 0
    gpu_ids = [0, ]
    node_id = 0
    node_ids = [0, ]
    dataset_params = dict(
		name = dataset_name,
		annFile = annFile,
        ds_type = 'eval',
	)

    name_format="nopad_png"


    set_logger(log_level='info', fname=os.path.join(workdir, 'output.log'))

    ds = create_dataset(
        gpu_id=gpu_id,
        gpu_ids=gpu_ids,
        node_id=node_id,
        node_ids=node_ids,
		**dataset_params,
	)

    real_image_path = ds.root if hasattr(ds, "root") else None

    n_images = len(ds.anno["annotations"])

    results = evaluate_quantitative_scores_text2img(
        pipe=None,
        real_image_path=real_image_path,
        mscoco_anno=ds.anno,
        n_images=n_images,
        batchsize=1,
        seed=1, 
        fake_image_path=workdir,
        name_format=name_format,
    )
    for k, v in results.items():
        logging.info(f"{k}: {v}")
