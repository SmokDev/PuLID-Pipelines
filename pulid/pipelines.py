from .core import PuLID, hack_unet_ca_layers
import torch
from .encoders import IDEncoder, IDFormer
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)
from typing import Type, Dict
from functools import wraps

def pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class PuLIDPipeline(pipeline_constructor):

        def load_pulid(self, 
            weights_or_pulid: PuLID | str | Dict[str, torch.Tensor],
            id_encoder: IDEncoder | IDFormer = None,
            use_id_former: bool = True
        ):
            is_pulid_instance = isinstance(weights_or_pulid, PuLID)

            if is_pulid_instance:
                self.pulid = PuLID(
                    id_encoder=weights_or_pulid.id_encoder,
                    features_extractor=weights_or_pulid.features_extractor,
                    ca_layers=hack_unet_ca_layers(self.unet)
                )
                self.pulid.ca_layers.load_state_dict(torch.nn.ModuleList(self.unet.attn_processors.values()).state_dict())
            else:
                if not hasattr(self, "pulid"):
                    self.pulid = PuLID(id_encoder=id_encoder, use_id_former=use_id_former, ca_layers=hack_unet_ca_layers(self.unet))
                self.pulid.load_weights(weights_or_pulid)

        def to(self, device: str):
            super().to(device)
            self.pulid.to(device)
        
        @classmethod
        @wraps(pipeline_constructor.from_pipe)
        def from_pipe(cls, pipeline, **kwargs):
            pipe = super().from_pipe(pipeline, **kwargs)
            if isinstance(pipeline, PuLIDPipeline):
                if hasattr(pipeline, "pulid"): pipe.load_pulid(pipeline.pulid)
            else: pipe = cls(**pipe.components)
            return pipe
        
        @wraps(pipeline_constructor.__call__)
        def __call__(self, *args,
            id_image = None,
            id_scale: float = 1,
            pulid_ortho: str = "off",
            pulid_editability: int = 16,
            pulid_mode:str = None,
            **kwargs
        ):
            pulid_cross_attention_kwargs = {}
            cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})

            if not pulid_mode == None:
                self.pulid.set_mode(pulid_mode)
            else:
                self.pulid.set_editability(pulid_editability)
                self.pulid.set_ortho(pulid_ortho)

            if not id_image == None:
                id_embedding = self.pulid(id_image)
                pulid_cross_attention_kwargs = { 'id_embedding': id_embedding, 'id_scale': id_scale }

            return super().__call__(*args, cross_attention_kwargs={**pulid_cross_attention_kwargs, **cross_attention_kwargs}, **kwargs )
        
    return PuLIDPipeline


# SDXL Pipelines
StableDiffusionXLPuLIDPipeline = pipeline_creator(StableDiffusionXLPipeline)
StableDiffusionXLPuLIDImg2ImgPipeline = pipeline_creator(StableDiffusionXLImg2ImgPipeline)
StableDiffusionXLPuLIDInpaintPipeline = pipeline_creator(StableDiffusionXLInpaintPipeline)
StableDiffusionXLPuLIDControlNetPipeline = pipeline_creator(StableDiffusionXLControlNetPipeline)
StableDiffusionXLPuLIDControlNetImg2ImgPipeline = pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)
StableDiffusionXLPuLIDControlNetInpaintPipeline = pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)

__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline"
]