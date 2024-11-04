import os
import random
from datetime import datetime

import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np

import einops
from PIL import Image

from .multi_gpu_dataframe_split import split_dataframe_for_gpu, split_dataframe_for_node, split_datalist_for_gpu

def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)
    return np.array(img).astype(np.uint8)

class PartiPromptsMultiGPUBench(Dataset):

    def __init__(
        self, 
        annFile, 
        gpu_id,
        gpu_ids,
        node_id,
        node_ids,
        output_dir=None,
    ):
        csv_file = annFile
        print(f"Loading PartiPrompts from {csv_file} for GPU {gpu_id}, Node {node_id}")
        self.df = pd.read_csv(csv_file, sep='\t')
        self.csv_file_base_name = os.path.basename(csv_file)

        self.not_name_char = [
            '\"', "'", "(", ")", ":", ";", ",", ".", 
            "!", "?", ">", "<", "[", "]", "{", "}", 
            "|", "\\", "/", "@", "#", "$", "%", "^",
            "&", "*", "~", "`", "=", "+", "-", "_",
        ]

        self.prompt_dict = self.check_all_prompts()

        self.df = split_dataframe_for_gpu(self.df, gpu_id, gpu_ids, node_id, node_ids)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        prompt = self.df.iloc[idx]['Prompt']
        # prompt = self.clean_prompt(prompt)
        prompt_idx = self.prompt_dict[ prompt ]

        return prompt, prompt_idx

    def clean_prompt(self, prompt):
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("\t", " ")
        prompt = prompt.replace("\r", " ")
        prompt = prompt.replace("  ", " ")
        prompt = prompt.strip()
        prompt = prompt.lower()
        for char in self.not_name_char:
            prompt = prompt.replace(char, " ")
        return prompt
    
    def check_all_prompts(self):
        prompt_dict = dict()
        max_len = 0
        for idx in range(len(self.df)):
            prompt = self.df.iloc[idx]['Prompt']
            # prompt = self.clean_prompt(prompt)
            prompt_dict[prompt] = idx
            max_len = max(max_len, len(prompt))
        
        print(f"Number of unique prompts: {len(prompt_dict)} | Max prompt length: {max_len}")
        return prompt_dict

class PartiPromptsMultiGPUBenchCOCOFormat(PartiPromptsMultiGPUBench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno = dict()
        self.anno["annotations"] = []
        for idx in range(len(self.df)):
            self.anno["annotations"].append({
                "id": idx,
                "caption": self.df.iloc[idx]['Prompt'],
            })

class MSCOCODatabase(Dataset):
    def __init__(
        self, 
        root='data/coco/val2017', 
        annFile='data/coco/annotations/captions_val2017.json',
        size=None,
        **kwargs,
    ):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size
        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')
        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target

class MSCOCOPromptBench(MSCOCODatabase):
    def __init__(
        self, 
        gpu_id,
        gpu_ids,
        node_id,
        node_ids,
        *args, 
        output_dir=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_coco_dataset_dict(gpu_id, gpu_ids, node_id, node_ids)
    
    def _init_coco_dataset_dict(self, gpu_id, gpu_ids, node_id, node_ids):
        keys = self.keys

        max_relative_id = 0
        max_prompt_len = 0
        self.anno = dict()
        self.anno["annotations"] = []
        for key in keys:
            target = []
            ids = []
            for i, ann in enumerate(self._load_target(key)):
                if max_prompt_len < len(ann['caption']):
                    max_prompt_len = len(ann['caption'])
                    max_relative_id = i
                
                target.append(ann['caption'])
                ids.append(ann['id'])
            
            prompt = target[max_relative_id]
            prompt_idx = ids[max_relative_id]

            self.anno["annotations"].append({
                "id": prompt_idx,
                "caption": prompt,
            })
        
        self.anno_dict_keys = list(range(len(self.anno["annotations"])))
        
        self.anno_dict_keys = split_datalist_for_gpu(
            self.anno_dict_keys, gpu_id, gpu_ids, node_id, node_ids
        )
    
    def __len__(self):
        return len(self.anno_dict_keys)
    
    def __getitem__(self, index):
        key = self.anno_dict_keys[index]

        prompt_dict = self.anno["annotations"][key]
        prompt = prompt_dict["caption"]
        prompt_idx = prompt_dict["id"]

        return prompt, prompt_idx

def create_dataset(
    name, 
    ds_type='eval',
    **kwargs,
):
    # train/test split datasets
    if ds_type == 'eval':
        if name == "coco":
            ds = MSCOCOPromptBench(**kwargs)
            return ds
        elif name == "parti_cocoformat":
            return PartiPromptsMultiGPUBenchCOCOFormat(**kwargs)
        elif name == "parti":
            return PartiPromptsMultiGPUBench(**kwargs)
        else:
            raise NotImplementedError
    else:
        if name == "coco":
            ds = MSCOCODatabase(**kwargs)
            return ds
        else:
            raise NotImplementedError
    
if __name__ == "__main__":
    ds = create_dataset(
        name="mscoco",
        root='data/coco/train2017', 
        annFile='data/coco/annotations/captions_val2017.json',
    )
    print(len(ds.anno["annotations"]))