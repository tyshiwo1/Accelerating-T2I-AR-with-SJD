import torch
from PIL import Image
import requests
import time
import os

import sys
import warnings
from functools import cached_property
from typing import Dict, Literal, Optional, Tuple, Union

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.generation.logits_process import (
    LogitsProcessor,
)

from scheduler.logit_processor_3dim import SequenceSegmentDecomposer, get_double_cfg_input_ids, \
    check_eol_in_multitokens, get_eol_in_multitokens

from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler, renew_backbone

def debug_attn_mask(attention_mask, name='emu3attention_mask.png'):
    from matplotlib import pyplot as plt
    if len(attention_mask.shape) == 4:
        attention_mask = attention_mask[0, 0, :, :]
    
    attention_mask = attention_mask.exp()

    attention_mask = attention_mask.float().cpu().numpy()
    plt.imshow(attention_mask, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Attention Mask')
    plt.savefig(name)
    plt.close()

def renew_end_of_line_logit_processor_3d(model_class):
    class EOLLogitProcessor3d(model_class, LogitsProcessor):
        
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            new_scores = []
            for batch_id in range(input_ids.shape[0]):
                batch_input_ids = input_ids[batch_id]
                batch_scores = scores[batch_id]

                if batch_id not in self.offset_cache:
                    position = torch.nonzero(batch_input_ids == self.img_token, as_tuple=True)[0][0]
                    self.offset_cache[batch_id] = position

                # offset = batch_input_ids.shape[0] - self.offset_cache[batch_id]

                pad_eol_len = 1
                new_logit_token_len = batch_scores.shape[-2] if len(batch_scores.shape) >= 2 else 1
                tokens = batch_input_ids[ self.offset_cache[batch_id] + 1 : ]

                is_new_seq_ids_containing_eol = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, self.width + pad_eol_len
                )
                is_new_seq_ids_containing_eof = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, 
                    (self.width + pad_eol_len) * self.height + 1
                )
                is_new_seq_ids_containing_eoi = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, 
                    (self.width + pad_eol_len) * self.height + 2
                )
                is_new_seq_ids_containing_eos = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, 
                    (self.width + pad_eol_len) * self.height + 3
                )
                is_new_seq_ids_containing_larger_eos = (
                    len(tokens) + new_logit_token_len > (
                        self.width + pad_eol_len) * self.height + 3
                )

                min_dtype = torch.finfo(batch_scores.dtype).min
                batch_scores = torch.full_like(batch_scores, min_dtype) # N, C
                batch_scores[:, self.visual_tokens] = scores[batch_id, :, self.visual_tokens]

                # containing ONE end-of-line
                if is_new_seq_ids_containing_eol:

                    batch_scores, eol_position_ids = get_eol_in_multitokens(
                        batch_scores, self.eol_token, len(tokens), 
                        new_logit_token_len, self.width + pad_eol_len,
                        min_dtype=min_dtype,
                    )
                
                # containing ONE end-of-image
                if is_new_seq_ids_containing_eof:
                    batch_scores, eol_position_ids = get_eol_in_multitokens(
                        batch_scores, self.eof_token, len(tokens), 
                        new_logit_token_len, 
                        (self.width + pad_eol_len) * self.height + 1,
                        min_dtype=min_dtype,
                    )
                
                if is_new_seq_ids_containing_eoi:
                    batch_scores, eol_position_ids = get_eol_in_multitokens(
                        batch_scores, self.eoi_token, len(tokens), 
                        new_logit_token_len, 
                        (self.width + pad_eol_len) * self.height + 2,
                        min_dtype=min_dtype,
                    )
                
                if is_new_seq_ids_containing_eos:
                    batch_scores, eol_position_ids = get_eol_in_multitokens(
                        batch_scores, self.eos_token, len(tokens), 
                        new_logit_token_len, 
                        (self.width + pad_eol_len) * self.height + 3,
                        min_dtype=min_dtype,
                    )
                
                if is_new_seq_ids_containing_larger_eos:
                    larger_eos_start_idx = (
                        (self.width + pad_eol_len) * self.height + 3 - len(tokens)
                    )
                    batch_scores[ larger_eos_start_idx:, :] = min_dtype
                    batch_scores[ larger_eos_start_idx:, self.pad_token ] = 0

                new_scores.append(batch_scores)
            
            scores = torch.stack(new_scores, dim=0)
            return scores


        def _call_in_beam_search(self, batch_id, input_ids):
            if batch_id not in self.offset_cache:
                position = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0][0]
                self.offset_cache[batch_id] = position

            offset = input_ids.shape[0] - self.offset_cache[batch_id]
            if offset % (self.width + 1) == 0:
                return (self.eol_token, )
            elif offset == (self.width + 1) * self.height + 1:
                return (self.eof_token, )
            elif offset == (self.width + 1) * self.height + 2:
                return (self.eoi_token, )
            elif offset == (self.width + 1) * self.height + 3:
                return (self.eos_token, )
            elif offset > (self.width + 1) * self.height + 3:
                return (self.pad_token, )
            else:
                return self.visual_tokens

    
    return EOLLogitProcessor3d

def renew_sampler_forward(model_class):
    class JacobiModel(model_class):
        def _init_new_params(
            self, 
            *args, 
            use_chameleon_tokenizer=False, 
            _init_doubled_attn_mask_cfg=True,
            visual_tokens=None,
            **kwargs,
        ):
            super()._init_new_params(
                *args, 
                use_chameleon_tokenizer=use_chameleon_tokenizer, 
                _init_doubled_attn_mask_cfg=_init_doubled_attn_mask_cfg,
                **kwargs,
            )
            if (not hasattr(self, 'img_vocab')) or (self.img_vocab is None):
                self.img_vocab = visual_tokens
                print('visual_tokens', visual_tokens)

            self._init_doubled_attn_mask_cfg = _init_doubled_attn_mask_cfg
            self.cfg_repeat_name_list = ['inputs_embeds', 'input_ids', 'pixel_values', ] 
            self.cfg_half_name_list = ['inputs_embeds', 'input_ids', 'pixel_values', ] 
        
        def renew_attn_mask(self, batchsize, prefill_num, not_pad_mask=None, device='cuda'):
            B_cfg = 2 * batchsize if self.do_cfg else batchsize
            attention_mask = torch.ones(
                (B_cfg, prefill_num),
                device=device,
            )
            
            attention_mask[~not_pad_mask] = 0
            
            return attention_mask

        # def _sample(self, input_ids=None, attention_mask=None, neg_input_ids=None, **kwargs):
        #     return super()._sample(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask, 
        #         neg_input_ids=neg_input_ids,
        #         **kwargs,
        #     )
        @torch.no_grad()
        def forward(
            self, 
            *args, 
            input_ids=None, 
            position_ids = None,
            past_key_values = None,
            attention_mask=None, 
            pixel_values=None, 
            cache_position=None, 
            output_attentions: Optional[bool] = None,
            **kwargs,
        ):

            dtype = self.dtype
            fake_inputs_embeds = torch.empty_like(input_ids, dtype=dtype)

            attention_mask = self._update_causal_mask(
                attention_mask, fake_inputs_embeds, cache_position, past_key_values, output_attentions
            )

            return super().forward(
                *args, input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                **kwargs,
            )
        
        def prepare_inputs_for_generation(self, *args, neg_input_ids=None, **kwargs):
            model_inputs = super().prepare_inputs_for_generation(
                *args,
                neg_input_ids=neg_input_ids,
                **kwargs,
            )
            model_inputs['neg_input_ids'] = neg_input_ids
            return model_inputs
        
        def prepare_batch_cfg_model_inputs(self, input_ids, neg_input_ids=None, attention_mask=None):
            # model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
            # attention_mask = model_inputs.get('attention_mask', None)
            # input_ids = model_inputs['input_ids']
            model_inputs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            batchsize, prefill_num = input_ids.shape

            neg_prefill_num = neg_input_ids.shape[1] if neg_input_ids is not None else prefill_num

            batchsize_cfg = 2 * batchsize if self.do_cfg else batchsize
            max_prefill_num = max(prefill_num, neg_prefill_num)
            not_pad_mask = torch.zeros(
                (batchsize_cfg, max_prefill_num),
                dtype=torch.bool,
                device=input_ids.device
            )
            not_pad_mask[:batchsize, -input_ids.shape[1]:] = input_ids != self.config.pad_token_id

            if neg_input_ids is not None:
                new_neg_input_ids = get_double_cfg_input_ids(
                    input_ids, 
                    neg_input_ids,
                    pad_category = self.config.pad_token_id,
                )

                model_inputs['input_ids'] = new_neg_input_ids
                model_inputs['pos_input_ids'] = new_neg_input_ids[:batchsize, :] # should be padded during preparation

                not_pad_mask[:, :] = new_neg_input_ids != self.config.pad_token_id

            if (attention_mask is None):
                attention_mask = self.renew_attn_mask(
                    batchsize=batchsize, 
                    prefill_num = max_prefill_num, 
                    not_pad_mask = not_pad_mask,
                    device=input_ids.device,
                )
                model_inputs['attention_mask'] = attention_mask
            elif attention_mask.shape[0] == batchsize:
                raise NotImplementedError

            return model_inputs
        
        def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            # # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
            #         attention_mask,
            #         inputs_embeds=input_tensor,
            #         past_key_values_length=past_seen_tokens,
            #         is_training=self.training,
            #     ):
            #         return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
                causal_mask = attention_mask
            else:
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)
                    
                    # debug_attn_mask(causal_mask[:, :, :, :mask_length], name='emu3precausalattention_mask.png')
                    # debug_attn_mask(attention_mask, name='emu3pre2attention_mask.png')

                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    # print(causal_mask, attention_mask, causal_mask.shape, attention_mask.shape)
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                    # debug_attn_mask(causal_mask, name='emu3causalattention_mask.png')
            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            # print('causal_mask', causal_mask.shape, causal_mask)
            return causal_mask
    
    return JacobiModel

def renew_solver(model, processor, **jacobi_param_dict):
    h = jacobi_param_dict.pop('h') if 'h' in jacobi_param_dict else None
    w = jacobi_param_dict.pop('w') if 'w' in jacobi_param_dict else None
    neg_input_ids = jacobi_param_dict.pop('neg_inputs') if 'neg_inputs' in jacobi_param_dict else None
    classifier_free_guidance = jacobi_param_dict.pop(
        'classifier_free_guidance') if 'classifier_free_guidance' in jacobi_param_dict else None

    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    constrained_fn.__class__ = renew_end_of_line_logit_processor_3d(constrained_fn.__class__)

    # model.__class__ = renew_backbone(model.__class__)
    model.__class__ = renew_sampler(model.__class__)
    model._init_new_params(**jacobi_param_dict)
    model.__class__ = renew_sampler_forward(model.__class__)
    model._init_new_params(
        visual_tokens=constrained_fn.visual_tokens, 
        **jacobi_param_dict,
    )

    # logits_processor = LogitsProcessorList([
    #     # UnbatchedClassifierFreeGuidanceLogitsProcessor(
    #     #     classifier_free_guidance,
    #     #     model,
    #     #     unconditional_ids=neg_input_ids,
    #     # ),
    #     PrefixConstrainedLogitsProcessor(
    #         constrained_fn ,
    #         num_beams=1,
    #     ),
    # ])
    logits_processor = LogitsProcessorList([
        constrained_fn,
    ])
    # logits_processor = LogitsProcessorList([
    #     SequenceSegmentDecomposer(
    #         sublogit_processors = logits_processor,
    #         do_sample = True,
    #         seed=None,
    #         fix_logits=False,
    #     ),
    # ])

    return model, logits_processor