import os
import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from torchvision import transforms

class PipelineLoader:
    def __init__(self, base_path: str, device: str = "cuda"):
        self.base_path = base_path
        self.device = device
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self._load_components()
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

        self.UNet_Encoder = UNet2DConditionModel.from_pretrained(
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

    def load_pipeline(self):
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

        return self.pipe