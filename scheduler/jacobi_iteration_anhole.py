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

from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler, renew_backbone

# from transformers import ChameleonPreTrainedModel

from .logit_processor_3dim import (
    AllowOnlyTokensAtRelativeOffsetLogitsProcessor3d, 
    SuppressTokensInIndexRangeLogitsProcessor3d,
    AllowOnlyTokensInRelativeWindowLogitsProcessor3d,
    SuppressTokensAtBeginLogitsProcessor3d,
    SuppressTokensLogitsProcessor3d,
)

# [
# <transformers.generation.logits_process.AllowOnlyTokensAtRelativeOffsetLogitsProcessor object at 0x738853307340>, 
# <transformers.generation.logits_process.AllowOnlyTokensInRelativeWindowLogitsProcessor object at 0x738853307c40>, 
# <transformers.generation.logits_process.SuppressTokensInIndexRangeLogitsProcessor object at x7388533076d0>, 
# <transformers.generation.logits_process.SuppressTokensLogitsProcessor object at 0x738853307580>, 
# <transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor object at 0x738853307010>, 
# <transformers.generation.logits_process.TopKLogitsWarper object at 0x73885a8a7010>
# ]

# [
# <scheduler.logit_processor_3dim.AllowOnlyTokensAtRelativeOffsetLogitsProcessor3d object at 0x7e24445526e0>, 
# <scheduler.logit_processor_3dim.AllowOnlyTokensInRelativeWindowLogitsProcessor3d object at 0x7e2444552710>, 
# <scheduler.logit_processor_3dim.SuppressTokensInIndexRangeLogitsProcessor3d object at 0x7e2444552830>, 
# <transformers.generation.logits_process.SuppressTokensLogitsProcessor object at 0x7e2444552800>, 
# <scheduler.logit_processor_3dim.SuppressTokensAtBeginLogitsProcessor3d object at 0x7e2444552920>, 
# <transformers.generation.logits_process.TopKLogitsWarper object at 0x7e2444552bc0>] None

def renew_vocabulary_mapping(vocabulary_mapping):
    class IndexVocabularyMapping(vocabulary_mapping):
        def _init_new_params(
            self,
            image_token_id: int = 8711,
            boi_token_id: int = 8197,
            eoi_token_id: int = 8196,
        ):
            if not hasattr(self, "image_token_id"):
                self.image_token_id = image_token_id

            if not hasattr(self, "boi_token_id"):
                self.boi_token_id = boi_token_id

            if not hasattr(self, "eoi_token_id"):
                self.eoi_token_id = eoi_token_id

        @cached_property
        def val2name(self):
            return {v: k for k, v in self.vocab_map.items()}

        @cached_property
        def image_token_ids(self):
            return sorted([val for name, val in self.vocab_map.items() if name.startswith("IMGIMG")])

        @cached_property
        def bpe2img(self):
            img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

            def remap(old_name: str) -> str:
                return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

            return {tok: int(remap(self.val2name[tok])) for tok in self.image_token_ids}

        @cached_property
        def img2bpe(self):
            return {v: k for k, v in self.bpe2img.items()}

        @cached_property
        def bpe2img_mapping_tensor(self):
            mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
            for k, v in self.bpe2img.items():
                mapping[k] = v
            return mapping

        @cached_property
        def img2bpe_mapping_tensor(self):
            mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
            for k, v in self.img2bpe.items():
                mapping[k] = v
            return mapping
    
    return IndexVocabularyMapping

def renew_pipeline_anole(model_class):
    class JacobiPipeline(model_class):
        def _init_new_params(self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

            self.vocabulary_mapping = self.model.vocabulary_mapping # ChameleonImageVocabularyMapping
            self.vocabulary_mapping.__class__ = renew_vocabulary_mapping(self.vocabulary_mapping.__class__)
            self.vocabulary_mapping._init_new_params()
        
        def _prepare_generation_config(
            self,
            generation_config = None,
            multimodal_generation_mode: Optional[
                Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]
            ] = None,
            **kwargs,
        ):
            if (
                multimodal_generation_mode == "image-only"
                and kwargs.get("max_length") is None
                and kwargs.get("max_new_tokens") is None
                and (
                    generation_config is None
                    or (generation_config.max_length is None and generation_config.max_new_tokens is None)
                )
            ):
                kwargs["max_new_tokens"] = self.model.image_seq_length + 2
            generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
            if multimodal_generation_mode is not None:
                generation_config.multimodal_generation_mode = multimodal_generation_mode
            if (
                not hasattr(generation_config, "multimodal_generation_mode")
                or generation_config.multimodal_generation_mode is None
            ):
                generation_config.multimodal_generation_mode = "text-only"
            return generation_config, model_kwargs
        
        @torch.no_grad()
        def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            multimodal_generation_mode: Optional[
                Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]
            ] = None,
            **kwargs,
        ):

            generation_config, model_kwargs = self._prepare_generation_config(
                generation_config, multimodal_generation_mode, **kwargs
            )

            inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
                inputs, generation_config.bos_token_id, model_kwargs
            )
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

            # Prepare `max_length` depending on other stopping criteria.
            input_ids_length = input_ids.shape[-1]
            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
            generation_config = self._prepare_generated_length(
                generation_config=generation_config,
                has_default_max_length=has_default_max_length,
                has_default_min_length=has_default_min_length,
                model_input_name=model_input_name,
                inputs_tensor=inputs_tensor,
                input_ids_length=input_ids_length,
            )

            if logits_processor is None:
                logits_processor = LogitsProcessorList()
            if generation_config.multimodal_generation_mode == "text-only":
                logits_processor.append(
                    SuppressTokensLogitsProcessor3d(
                        suppress_tokens=self.vocabulary_mapping.image_token_ids
                        + [
                            self.vocabulary_mapping.boi_token_id,
                            self.vocabulary_mapping.eoi_token_id,
                        ],
                        device=self.device,
                    )
                )
            elif generation_config.multimodal_generation_mode == "image-only":
                inferred_max_new_tokens = generation_config.max_length - input_ids_length
                if inferred_max_new_tokens < self.model.image_seq_length + 2:
                    warnings.warn(
                        f"The VQVAE decoder expects to receive {self.model.image_seq_length} image tokens to generate an image."
                        "And Chameleon wraps the image tokens with the `beginning-of-image` and `end-of-image` tokens when on image generation mode."
                        f"Therefore, the `max_new_tokens` must be at least {self.model.image_seq_length + 2}."
                        f"However, the inferred `max_new_tokens` from the generation config is only {inferred_max_new_tokens}."
                        "You would need to pad the output tokens with dummy image tokens before passing them to the VQVAE decoder."
                        f"To avoid this warning, set `max_new_tokens` to at least {self.model.image_seq_length + 2}."
                    )
                allowed_tokens = self.vocabulary_mapping.image_token_ids + [
                    self.config.eos_token_id,
                    self.vocabulary_mapping.boi_token_id,
                    self.vocabulary_mapping.eoi_token_id,
                ]
                suppress_tokens = [token_id for token_id in range(self.vocab_size) if token_id not in allowed_tokens]
                logits_processor.extend(
                    [
                        AllowOnlyTokensAtRelativeOffsetLogitsProcessor3d(
                            trigger_token_id=self.vocabulary_mapping.boi_token_id,
                            allowed_token_ids=[self.vocabulary_mapping.eoi_token_id],
                            offset=self.model.image_seq_length + 1,
                            exclusive=True,
                            device=self.device,
                        ),
                        AllowOnlyTokensInRelativeWindowLogitsProcessor3d(
                            trigger_token_id=self.vocabulary_mapping.boi_token_id,
                            allowed_token_ids=self.vocabulary_mapping.image_token_ids,
                            window_width=self.model.image_seq_length,
                            exclusive=True,
                            device=self.device,
                        ),
                        # Don't start generating an image if there aren't enough space for the
                        # rest of the image tokens.
                        SuppressTokensInIndexRangeLogitsProcessor3d(
                            suppress_tokens=[self.vocabulary_mapping.boi_token_id],
                            start_index=generation_config.max_length - self.model.image_seq_length - 1,
                            device=self.device,
                        ),
                        # Allow only image tokens
                        SuppressTokensLogitsProcessor3d(suppress_tokens=suppress_tokens, device=self.device),
                        # Force image generation
                        SuppressTokensAtBeginLogitsProcessor3d(
                            begin_suppress_tokens=[self.config.eos_token_id],
                            begin_index=input_ids_length,
                            device=self.device,
                        ),
                    ]
                )
            elif generation_config.multimodal_generation_mode == "interleaved-text-image":
                logits_processor.extend(
                    [
                        AllowOnlyTokensAtRelativeOffsetLogitsProcessor3d(
                            trigger_token_id=self.vocabulary_mapping.boi_token_id,
                            allowed_token_ids=[self.vocabulary_mapping.eoi_token_id],
                            offset=self.model.image_seq_length + 1,
                            exclusive=True,
                            device=self.device,
                        ),
                        AllowOnlyTokensInRelativeWindowLogitsProcessor3d(
                            trigger_token_id=self.vocabulary_mapping.boi_token_id,
                            allowed_token_ids=self.vocabulary_mapping.image_token_ids,
                            window_width=self.model.image_seq_length,
                            exclusive=True,
                            device=self.device,
                        ),
                        # Don't start generating an image if there aren't enough space for the
                        # rest of the image tokens.
                        SuppressTokensInIndexRangeLogitsProcessor3d(
                            suppress_tokens=[self.vocabulary_mapping.boi_token_id],
                            start_index=generation_config.max_length - self.model.image_seq_length - 1,
                            device=self.device,
                        ),
                    ]
                )
            elif generation_config.multimodal_generation_mode == "unrestricted":
                pass
            else:
                raise ValueError(
                    f"Unknown multimodal generation mode: {generation_config.multimodal_generation_mode}. Please choose one of 'unrestricted', 'text-only', 'image-only', or 'interleaved-text-image'."
                )
            
            generation_config.multimodal_generation_mode = "unrestricted"
            return super().generate(
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                **kwargs,
            )
        
        def decode_image_tokens(self, bpe_tokens: torch.Tensor):
            """
            Converts BPE tokens generated by the model into discrete image tokens
            compatible with the VQGAN module, then decodes them into pixel values.

            Args:
                bpe_tokens (`torch.tensor` of shape `(batch, image_seq_length)`):
                    The BPE tokens generated by the model.

            Returns:
                `torch.Tensor` of shape `(batch, num_channels, 512, 512)`:
            """
            return self.model.decode_image_tokens(bpe_tokens)
    
    return JacobiPipeline

def renew_backbone_adapt_anole(model_class):
    class JacobiBackboneAdaptedAnole(model_class):

        # @property
        # def image_seq_length(self) -> int: 
        #     return self.vqmodel.quantize.quant_state_dims[0] * self.vqmodel.quantize.quant_state_dims[1]
        def decode_image_tokens(self, bpe_tokens: torch.LongTensor) -> torch.LongTensor:
            """
            Converts BPE tokens generated by the model into discrete image tokens
            compatible with the VQGAN module, then decodes them into pixel values.

            Args:
                bpe_tokens (`torch.tensor` of shape `(batch, image_seq_length)`):
                    The BPE tokens generated by the model.

            Returns:
                `torch.Tensor` of shape `(batch, num_channels, 512, 512)`:
            """
            # print(bpe_tokens.shape, self.image_seq_length)
            if bpe_tokens.shape[1] != self.image_seq_length:
                # raise ValueError(f"All batches must have {self.image_seq_length} tokens.")
                bpe_tokens = bpe_tokens[:, :self.image_seq_length]

            image_tensor = self.convert_bpe2img_tokens(bpe_tokens)
            return self.vqmodel.decode(image_tensor)
    
    return JacobiBackboneAdaptedAnole

def renew_pipeline_sampler(pipe_line, processor, **kwargs):
    pipe_line.model.__class__ = renew_backbone(pipe_line.model.__class__)
    pipe_line.model.__class__ = renew_backbone_adapt_anole(pipe_line.model.__class__)
    if not hasattr(pipe_line.model, "image_seq_length"):
        pipe_line.model.image_seq_length = processor.image_seq_length
    
    print(pipe_line.model.image_seq_length)
    
    pipe_line.__class__ = renew_pipeline_anole(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    pipe_line.__class__ = renew_sampler(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    return pipe_line

