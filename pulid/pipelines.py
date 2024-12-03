from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)

from diffusers import DiffusionPipeline
from diffusers.loaders.unet_loader_utils import  _maybe_expand_lora_scales
from diffusers.models.attention_processor import  (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0
    )

import torch
from typing import Type, Dict, get_type_hints
from functools import wraps

from .core import PuLIDEncoder, hack_unet, get_unet_attn_layers
from .attention_processors import PuLIDAttnProcessor, AttnProcessor
from .utils import load_file_weights, state_dict_extract_names


def pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class PuLIDPipeline(pipeline_constructor):

        pulid_encoder: PuLIDEncoder = None
        pulid_timestep_to_start: int = None
        avalible_pulid: bool = False

        def load_pulid(self, 
            weights: str | Dict[str, torch.Tensor],
            pulid_encoder: PuLIDEncoder = None,
            use_id_former: bool = True
        ):
            self.unet = hack_unet(self.unet)

            pulid_encoder = PuLIDEncoder(use_id_former=use_id_former) if pulid_encoder is None else pulid_encoder
            pulid_encoder.to(self.device)
            self.pulid_encoder = pulid_encoder

            state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
            state_dict = state_dict_extract_names(state_dict)  
            for module in state_dict:
                if module == "id_adapter" or module == "pulid_encoder":
                    self.pulid_encoder.id_encoder.load_state_dict(state_dict=state_dict[module], strict=False)
                elif module == "id_adapter_attn_layers" or module == "pulid_ca":
                    pulid_attn_layers = get_unet_attn_layers(self.unet)
                    pulid_attn_layers.load_state_dict(state_dict=state_dict[module], strict=False)


        def to(self, device: str):
            super().to(device)
            if hasattr(self, "pulid_encoder"):
                self.pulid_encoder.to(device)
        
        @classmethod
        @wraps(pipeline_constructor.from_pipe)
        def from_pipe(cls, pipeline, **kwargs):
            pipe = super().from_pipe(pipeline, **kwargs)
            if isinstance(pipeline, PuLIDPipeline):
                if hasattr(pipeline, "pulid_encoder"): pipe.pulid_encoder(pipeline.pulid_encoder)
            else: pipe = cls(**pipe.components)
            return pipe
        
        @wraps(pipeline_constructor.__call__)
        def __call__(self, *args,
            id_image = None,
            id_scale: float = 1,
            pulid_ortho: str = None,
            pulid_editability: int = 16,
            pulid_mode:str = None,
            pulid_timestep_to_start: int = 2,
            **kwargs
        ):
            pulid_cross_attention_kwargs = {}
            cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})
            user_step_callback = kwargs.pop("callback_on_step_end", None)
            step_callback = None

            self.pulid_timestep_to_start = pulid_timestep_to_start
            if self.pulid_timestep_to_start > 0:
                self.avalible_pulid = False
                def pulid_step_callback(self, step, timestep, callback_kwargs):
                    if self.pulid_timestep_to_start >=  step - 1:
                        self.avalible_pulid = True

                    if not user_step_callback == None:
                        return user_step_callback(self, step, timestep, callback_kwargs)
                    else: return callback_kwargs
                    
                step_callback = pulid_step_callback
            else:
                self.avalible_pulid = True
                step_callback = user_step_callback


            if not id_image == None:
                id_embedding = self.pulid_encoder(id_image)
                pulid_cross_attention_kwargs = {
                    'id_embedding': id_embedding,
                    'id_scale': id_scale,
                    'pulid_mode': pulid_mode,
                    'pulid_num_zero': pulid_editability,
                    'pulid_ortho': pulid_ortho,
                    'avalible_pulid': self.avalible_pulid
                }

            return super().__call__(
                *args,
                cross_attention_kwargs={**pulid_cross_attention_kwargs, **cross_attention_kwargs},
                callback_on_step_end=step_callback,
                **kwargs
            )
        
        @wraps(pipeline_constructor.set_ip_adapter_scale)
        def set_ip_adapter_scale(self, scale):
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            if not isinstance(scale, list):
                scale = [scale]
            scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

            for attn_name, attn_processor in unet.attn_processors.items():
                
                if isinstance(
                    attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                ) or (
                    isinstance(attn_processor, PuLIDAttnProcessor) and isinstance(attn_processor.original_attn_processor, (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    )
                ): 
                    attn_processor = attn_processor.original_attn_processor if isinstance(
                        attn_processor, PuLIDAttnProcessor
                    ) and isinstance(
                        attn_processor.original_attn_processor, (
                            IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
                        )
                    ) else attn_processor
                    if len(scale_configs) != len(attn_processor.scale):
                        raise ValueError(
                            f"Cannot assign {len(scale_configs)} scale_configs to "
                            f"{len(attn_processor.scale)} IP-Adapter."
                        )
                    elif len(scale_configs) == 1:
                        scale_configs = scale_configs * len(attn_processor.scale)
                    for i, scale_config in enumerate(scale_configs):
                        if isinstance(scale_config, dict):
                            for k, s in scale_config.items():
                                if attn_name.startswith(k):
                                    attn_processor.scale[i] = s
                        else:
                            attn_processor.scale[i] = scale_config

        @wraps(pipeline_constructor.unload_ip_adapter)
        def unload_ip_adapter(self):
            # remove CLIP image encoder
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
                self.image_encoder = None
                self.register_to_config(image_encoder=[None, None])

            # remove feature extractor only when safety_checker is None as safety_checker uses
            # the feature_extractor later
            if not hasattr(self, "safety_checker"):
                if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                    self.feature_extractor = None
                    self.register_to_config(feature_extractor=[None, None])

            # remove hidden encoder
            self.unet.encoder_hid_proj = None
            self.unet.config.encoder_hid_dim_type = None

            # Kolors: restore `encoder_hid_proj` with `text_encoder_hid_proj`
            if hasattr(self.unet, "text_encoder_hid_proj") and self.unet.text_encoder_hid_proj is not None:
                self.unet.encoder_hid_proj = self.unet.text_encoder_hid_proj
                self.unet.text_encoder_hid_proj = None
                self.unet.config.encoder_hid_dim_type = "text_proj"

            # restore original Unet attention processors layers
            attn_procs = {}
            for name, attn_processor in self.unet.attn_processors.items():

                if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                    attn_procs[name] = AttnProcessor
                elif isinstance(attn_processor, PuLIDAttnProcessor) and isinstance(attn_processor.original_attn_processor, (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    ):
                    attn_processor.original_attn_processor = AttnProcessor
                    attn_procs[name] = attn_processor

            self.unet.set_attn_processor(attn_procs)

    PuLIDPipeline.__call__.__annotations__ = {**get_type_hints(pipeline_constructor.__call__), **{
        'id_image': None,
        'id_scale': float,
        'pulid_ortho': str,
        'pulid_editability': int,
        'pulid_mode': str,
        'pulid_timestep_to_start': int,
    }}
        
    return PuLIDPipeline



# SDXL Pipelines
class StableDiffusionXLPuLIDPipeline(pipeline_creator(StableDiffusionXLPipeline)): pass
class StableDiffusionXLPuLIDImg2ImgPipeline(pipeline_creator(StableDiffusionXLImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDInpaintPipeline(pipeline_creator(StableDiffusionXLInpaintPipeline)): pass
class StableDiffusionXLPuLIDControlNetPipeline(pipeline_creator(StableDiffusionXLControlNetPipeline)): pass
class StableDiffusionXLPuLIDControlNetImg2ImgPipeline(pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDControlNetInpaintPipeline(pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)): pass

__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline"
]