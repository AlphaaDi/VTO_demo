import torch
from transformers import AutoTokenizer

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from torchvision import transforms


class VtoCore:
    def __init__(self, config):
        self.base_path = config['base_path']
        self.config = config
        self.device = config['device']

        self._load_vto_core()
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

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.pipe.unet_encoder = self.UNet_Encoder

    def get_tensor_transform(self):
        return self.tensor_transform

    def to(self, device):
        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)
        self.device = device

    def _load_vto_core(self):
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

    def encode_prompts(self, garment_des, attributes={}):
        attributes['garment_des'] = garment_des
        prompt = self.config["prompt"].format(**attributes)

        gender = attributes["gender"]
        additional_neg_prompt = self.config["negative_prompt_gender"][gender]
        negative_prompt = additional_neg_prompt + self.config["negative_prompt"]
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        prompt = f"a photo of {garment_des}"
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt]
        with torch.inference_mode():
            (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            prompt_embeds_c,
        )
    
    def prepare_images_for_model(self, pose_img, garm_img):
        pose_img_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
        garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(self.device, torch.float16)
        return pose_img_tensor, garm_tensor
    
    def process_vto_pipeline(
            self, 
            pose_img,
            garm_img,
            human_img,
            garment_des,
            mask,
            denoise_steps,
            seed,
            attributes,
        ):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Encode prompts
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    prompt_embeds_c,
                ) = self.encode_prompts(garment_des,attributes=attributes)

                # Prepare images for the model
                pose_img_tensor, garm_tensor = self.prepare_images_for_model(pose_img, garm_img)

                generator = (
                    torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                )

                # Generate images with the model
                pipe_result = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_tensor,
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )
        return pipe_result[0][0]