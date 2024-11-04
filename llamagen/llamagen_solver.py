import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

import torch._dynamo.config
import torch._inductor.config
import copy

from typing import Optional, Tuple

import transformers
from transformers.generation.utils import GenerationMixin

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
from transformers.generation.logits_process import TopKLogitsWarper
from transformers import GenerationConfig

from transformers.utils import ModelOutput
from dataclasses import dataclass

from transformers import StoppingCriteria, StoppingCriteriaList

from transformers.cache_utils import Cache, DynamicCache

from scheduler.logit_processor_3dim import TopPLogitsWarper3d

@dataclass
class BackboneOutput(ModelOutput):
    logits: torch.Tensor = None
    past_key_values: Cache = None

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits = model.inference(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits = model.inference(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits = model.inference(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model.inference(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]

def renew_llamagen(
    model_class,
):
    class WrappedLLamaGen(model_class, GenerationMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def _init_new_params(self, *args, **kwargs):
            self.config.is_encoder_decoder = False
        
        def clear_kvcache(self):
            for layer_idx, b in enumerate(self.layers):
                b.attention.kv_cache.k_cache[..., :, :] = 0
                b.attention.kv_cache.v_cache[..., :, :] = 0
        
        def assign_kvcache(self, past_key_values):
            for layer_idx, b in enumerate(self.layers):
                used_len = past_key_values.key_cache[layer_idx].shape[-2]
                b.attention.kv_cache.k_cache[..., :used_len, :] = past_key_values.key_cache[layer_idx]
                b.attention.kv_cache.v_cache[..., :used_len, :] = past_key_values.value_cache[layer_idx]
        
        def get_max_kvcache_len(self):
            max_kvcache_len = 0
            for b in self.layers:
                max_kvcache_len = max(max_kvcache_len, b.attention.kv_cache.k_cache.shape[-2])
            return max_kvcache_len
        
        def assign_past_key_values(self, past_key_values, used_len):
            for layer_idx, b in enumerate(self.layers):
                if layer_idx < len(past_key_values.key_cache):
                    past_key_values.key_cache[layer_idx] = b.attention.kv_cache.k_cache[..., :used_len, :]
                    past_key_values.value_cache[layer_idx] = b.attention.kv_cache.v_cache[..., :used_len, :]
                else:
                    past_key_values.key_cache.append(b.attention.kv_cache.k_cache[..., :used_len, :])
                    past_key_values.value_cache.append(b.attention.kv_cache.v_cache[..., :used_len, :])

            return past_key_values

        def forward(
            self, 
            input_ids,
            position_ids,
            cache_position,
            past_key_values,
            use_cache,
            attention_mask,
            **kwargs,
        ):
            dtype = self.tok_embeddings.weight.dtype

            input_pos = position_ids[0][-input_ids.shape[1]:]
            
            while attention_mask.dim() < 4:
                attention_mask = attention_mask.unsqueeze(1)
            
            max_kvcache_len = self.get_max_kvcache_len()
            if attention_mask.shape[-1] < max_kvcache_len:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.zeros(
                        *attention_mask.shape[:-1], 
                        max_kvcache_len - attention_mask.shape[-1],
                        dtype=attention_mask.dtype, 
                        device=attention_mask.device
                    )
                ], dim=-1)
            
            causal_mask = self.causal_mask
            while causal_mask.dim() < 4:
                causal_mask = causal_mask.unsqueeze(1)
            
            causal_mask = causal_mask[:, :, input_pos, :].to(attention_mask.dtype)
            attention_mask = torch.minimum(attention_mask, causal_mask)
            
            min_dtype = torch.finfo(dtype).min
            attention_mask = ((attention_mask == 0).to(dtype) * min_dtype).to(dtype)
            mask = attention_mask

            is_kvcache_not_empty = (past_key_values.get_seq_length() > 0)

            # idx = input_ids if is_kvcache_not_empty else None
            # cond_idx = None if is_kvcache_not_empty else input_ids
            idx = input_ids

            if is_kvcache_not_empty:
                self.assign_kvcache(past_key_values)
            
            logits = self.inference(
               idx = idx, 
               cond_idx = None,
               input_pos=input_pos, 
               mask=None, #mask, 
            )
            used_len = position_ids[0][-1:] + 1
            past_key_values = self.assign_past_key_values(past_key_values, used_len)
            outputs = BackboneOutput(
                logits = logits,
                past_key_values = past_key_values,
            )
            return outputs
        
        def inference(
            self, 
            idx: torch.Tensor, 
            cond_idx: torch.Tensor,  # cond_idx_or_embed
            input_pos:  Optional[torch.Tensor] = None, 
            mask: Optional[torch.Tensor] = None,
        ):
            if idx is not None and cond_idx is not None: # training or naive inference
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                token_embeddings = self.tok_embeddings(idx)
                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
                h = self.tok_dropout(token_embeddings)
                self.freqs_cis = self.freqs_cis.to(h.device)
            else:
                if cond_idx is not None: # pre fill in inference
                    token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                else: # decode_n_tokens(kv cache) in inference
                    token_embeddings = self.tok_embeddings(idx)
                
                bs = token_embeddings.shape[0]
                if mask is None:
                    mask = self.causal_mask[:bs, None, input_pos, :]
                h = self.tok_dropout(token_embeddings)
                self.freqs_cis = self.freqs_cis
            
            if self.training:
                freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            else:
                freqs_cis = self.freqs_cis[input_pos]
            # transformer blocks
            for layer in self.layers:
                h = layer(h, freqs_cis, input_pos, mask)
            
            # output layers
            h = self.norm(h)
            logits = self.output(h).float()
            
            if self.training:
                logits = logits[:, self.cls_token_num - 1:].contiguous()

            return logits
    
    return WrappedLLamaGen

class MaxlenCriteria(StoppingCriteria):
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
    
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.max_seq_length

class LlamaGenSolver:
    def __init__(self, model, image_top_k, image_top_p):
        self.model = model
        self.image_top_k = image_top_k
        self.image_top_p = image_top_p
    
    def _sample(self, *args, **kwargs):
        raise NotImplementedError
    
    def _init_model_kwargs(self, prefill_num, mask=None, device='cuda', *args, **kwargs):
        # print('mas22k', mask[0, :50, :50], mask.shape)
        model_kwargs = dict(
            use_cache = True,
            attention_mask = mask[0, -1:, :prefill_num] if mask is not None else torch.ones(
                (1, prefill_num), device=device
            ),
            past_key_values = transformers.DynamicCache(),
            cache_position=prefill_num,
        )
        return model_kwargs
    
    @torch.no_grad()
    def generate(self, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
        model = self.model
        if model.model_type == 'c2i':
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond) * model.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        elif model.model_type == 't2i':
            if cfg_scale > 1.0:
                cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = cond.shape[1]      
        else:
            raise Exception("please check model type")

        T_new = T + max_new_tokens
        max_seq_length = T_new
        max_batch_size = cond.shape[0]

        device = cond.device
        with torch.device(device):
            max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
            model.setup_caches(
                max_batch_size=max_batch_size_cfg, 
                max_seq_length=max_seq_length , 
                dtype=model.tok_embeddings.weight.dtype
            )
        
        if emb_masks is not None:
            assert emb_masks.shape[0] == max_batch_size
            assert emb_masks.shape[-1] == T
            if cfg_scale > 1.0:
                model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
            else:
                model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

            eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
            model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

        mask = model.causal_mask

        input_pos = torch.arange(0, T, device=device)
        next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)

        input_ids = next_token

        max_gen_len = max_seq_length # max_new_tokens
        temperature = 1.0
        synced_gpus = False
        stopping_criteria = StoppingCriteriaList([
            MaxlenCriteria(max_new_tokens) # As llamagen has T5
        ])
        generation_config = GenerationConfig(
            max_new_tokens=max_gen_len,
            max_length=max_gen_len,
            temperature=temperature,
            top_k=None,
            do_sample=True, # eos_token_id= [8710],
            _pad_token_tensor=None,
            return_dict_in_generate = False,
        )
        model_kwargs = self._init_model_kwargs(
            prefill_num = T+1, # existing `next_token`, an already decoded image token
            mask = None, # this mask only used for generating the positional indexes
            device = device,
        )
        logits_processor = self.create_logits_processor()

        outputs = model._sample(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria = stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=None,
            logits_warper=None,
            **model_kwargs,
        )
        generated_tokens = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
        generated_tokens = generated_tokens[:, -max_new_tokens:]
        model.clear_kvcache()
        return generated_tokens
    
    def create_logits_processor(self, ):
        image_top_k = self.image_top_k
        image_top_p = self.image_top_p

        logits_processor = LogitsProcessorList()

        topk_logits_warper = TopKLogitsWarper(top_k=image_top_k)
        topp_logits_warper = TopPLogitsWarper3d(top_p=image_top_p)

        logits_processor.append(topk_logits_warper)
        logits_processor.append(topp_logits_warper)

        return logits_processor