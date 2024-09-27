from PIL import Image

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from dw_pose import DWposeDetector
import torch

class StableDiffusionInpaintWrapper:
    def __init__(
        self, 
        model_path,
        dw_model_pathdir,
        device, 
        neg_prompt, 
        inpaint_timestep_num=50,
        guidance_scale = 4.0,
        strength = 0.99,
        max_timestep = 300,
        ip_adapter_scale = 0.7,
        controlnet_conditioning_scale = 0.7,
    ):
        self.device = device

        self.dw_pose = DWposeDetector(dir_path=dw_model_pathdir)
        self.dw_pose.to(device)

        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        )

        self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        self.inpaint_pipe.to('cuda:1')

        self.inpaint_pipe.load_ip_adapter(
            "ozzygt/sdxl-ip-adapter", "",
            "ip-adapter-plus_sdxl_vit-h.safetensors"
        )
        
        # Default negative prompt if none provided
        self.neg_prompt = neg_prompt
        self.inpaint_timestep_num = inpaint_timestep_num
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.max_timestep = max_timestep
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # Set scheduler timesteps
        self.inpaint_pipe.scheduler.config.num_train_timesteps = self.max_timestep
        self.inpaint_pipe.to(self.device)

        self.inpaint_pipe.set_ip_adapter_scale(ip_adapter_scale)


    def forward(self, image, mask, pos_prompt, ip_adapter_image):
        """
        Perform inpainting on the given image and mask.

        Args:
        - image (PIL.Image): The input image to be inpainted.
        - mask (PIL.Image): The mask indicating regions to inpaint.
        - pos_prompt (str): Positive prompt describing what should be generated.
        
        Returns:
        - PIL.Image: The inpainted image.
        """
        pose_img, _ = self.dw_pose(ip_adapter_image)
        pose_img_res = pose_img.resize(image.size)

        inpaint_missmatched_result = self.inpaint_pipe(
            prompt=pos_prompt,
            image=image,
            mask_image=mask,
            negative_prompt=self.neg_prompt,
            strength=self.strength,
            num_inference_steps=self.inpaint_timestep_num,
            guidance_scale=self.guidance_scale,
            control_image=pose_img_res,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            ip_adapter_image=ip_adapter_image,
        ).images[0]

        return inpaint_missmatched_result