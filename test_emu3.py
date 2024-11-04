import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
print(sys.path)

import os
import time


from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch

from emu3.mllm.processing_emu3 import Emu3Processor

cache_dir = "./ckpts/"
device = "cuda:0"

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

model_name = EMU_HUB.split("/")[-1]

dtype = torch.bfloat16

a = time.time()

# prepare model and processor
model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    device_map=device,
    torch_dtype=dtype,
    attn_implementation="sdpa", # "sdpa" , "flash_attention_2"
    trust_remote_code=True,
    cache_dir = cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, cache_dir=cache_dir,)
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, cache_dir=cache_dir,)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=device, trust_remote_code=True, cache_dir=cache_dir,).eval()

print(f"Time: {time.time() - a}")
a = time.time()

image_tokenizer = image_tokenizer.to(dtype)

processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

print(f"Time: {time.time() - a}")
a = time.time()

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
# prompt = "a portrait of young girl."
# prompt = "Portrait of an ancient Chinese boy with big eyes, muscular body, head lowered arrogantly, can use fire magic, with red sky in the background, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8,  ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, 8k."
# prompt = "A black bear with a kasaya (the robes of fully ordained Buddhist monks) around its waist. This black bear has a string of mala beads hanging diagonally on its body."
# prompt = "An oil painting of a lady"
# prompt = "A fantasy image of a person whose head is replaced by a blood-red hand wearing ancient Chinese armor"
# prompt = "A red monster with a head shaped like a human's hand (five fingers, no facial features), wearing ancient Chinese armor on his body, holding a sword with red lightning"
# prompt = "A giant black stone Macaque with thick black hair and muscles covered in black stone. As tall as a snowy mountain, with clouds and mist at its feet."
# prompt = "An imaginary picture of a hamster whose action is similar to human with one paw grasping a trident"
# prompt = "An imaginary picture of a warrior with his head replaced by a football"
prompt = "A fantastic imaginary picture of a chinese ancient general in full armor standing on a one-wheel cycle with red fire and lightning"
prompt += POSITIVE_PROMPT

kwargs = dict(
    mode='G',
    ratio="1:1",
    image_area=model.config.image_area,
    return_tensors="pt",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

image_top_k = 2048
# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=image_top_k,
)

print(f"Time: {time.time() - a}")
a = time.time()

h, w = pos_inputs.image_size[0]
print(h, w)
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to(device),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn ,
        num_beams=1,
    ),
])

def get_jacobi_param_dict(image_top_k):
    target_size = 720

    seeds = [None, ]
    max_num_new_tokens = 16 
    multi_token_init_scheme = 'random'
    text_top_k = 10
    guidance_scale = classifier_free_guidance
    prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 8)**2 -1,
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg= True, #True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
    )
    return jacobi_param_dict

pos_input_ids = pos_inputs.input_ids.to(device)
neg_input_ids = neg_inputs.input_ids.to(device)

jacobi_param_dict = get_jacobi_param_dict(
    image_top_k=image_top_k,
)
jacobi_param_dict['h'] = h
jacobi_param_dict['w'] = w
jacobi_param_dict['neg_inputs'] = neg_input_ids
jacobi_param_dict['classifier_free_guidance'] = classifier_free_guidance

from scheduler.jacobi_iteration_emu3 import renew_solver
model, logits_processor = renew_solver(model, processor, **jacobi_param_dict)
print(f"Time: {time.time() - a}")
a = time.time()
# generate
model_inputs = model.prepare_batch_cfg_model_inputs(
    pos_input_ids, 
    neg_input_ids=neg_input_ids, 
    attention_mask=None,
)
pos_input_ids = model_inputs['pos_input_ids']
attention_mask = model_inputs['attention_mask']

print(pos_input_ids.shape, neg_input_ids.shape)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            pos_input_ids,
            GENERATION_CONFIG,
            logits_processor=logits_processor, 
            attention_mask=attention_mask,
            neg_input_ids=neg_input_ids,
        )

outputs = outputs[0]
torch.save(outputs, "outputs_emu3.pt")


time_gen = time.time() - a

print(f"Time: {time_gen}", h, w) # Time: 512.5784916877747

output_path = f"./workdir/{model_name}" 
if not os.path.exists(output_path):
    os.makedirs(output_path)


print(outputs, outputs.shape, outputs.dtype)
with torch.no_grad():
    mm_list = processor.decode(outputs)

for idx, im in enumerate(mm_list):
    if not isinstance(im, Image.Image):
        continue
    im.save(
        os.path.join(
            output_path,
            f"result_{idx}.png"
        )
    )