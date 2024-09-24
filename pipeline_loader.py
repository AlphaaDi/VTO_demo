import torch
from transformers import AutoTokenizer
import numpy as np

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from torchvision import transforms

class PipelineLoader:
    def __init__(self, base_path: str, config: str, device: str = "cuda"):
        self.base_path = base_path
        self.device = device
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self._load_components()
        self._load_pipeline()
        self._move_to_device()

    def _load_components(self):
        # Load models
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.unet.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        self.vae.requires_grad_(False)

        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )
        self.UNet_Encoder.requires_grad_(False)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.image_encoder.requires_grad_(False)

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_one.requires_grad_(False)

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two.requires_grad_(False)

        # Load tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )

        # Load scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_path,
            subfolder="scheduler"
        )

        # Load parsing and openpose models
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

    def _move_to_device(self):
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

    def _load_pipeline(self):
        # Load pipeline
        self.pipe = TryonPipeline.from_pretrained(
            self.base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
        )

        # Attach the unet encoder to the pipeline
        self.pipe.unet_encoder = self.UNet_Encoder

    def get_pipeline(self):
        if not hasattr(self, 'pipe') or not self.pipe:
            raise AttributeError('Loader could not load pipeline, try again')
        return self.pipe

    def get_openpose_model(self):
        if not hasattr(self, 'openpose_model') or not self.pipe:
            raise AttributeError('Loader could not load openpose model, try again')
        return self.openpose_model

    def get_parsing_model(self):
        if not hasattr(self, 'parsing_model') or not self.pipe:
            raise AttributeError('Loader could not load parsing model, try again')
        return self.parsing_model
    
    def get_tensor_transform(self):
        if not hasattr(self, 'tensor_transform') or not self.pipe:
            raise AttributeError('Loader could not load tensor transform, try again')
        return self.tensor_transform
