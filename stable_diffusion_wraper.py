import torch
from diffusers import StableDiffusionXLInpaintPipeline

class StableDiffusionInpaintWrapper:
    def __init__(
        self, 
        model_path, 
        device='cpu', 
        dtype=torch.float16, 
        neg_prompt="", 
        inpaint_timestep_num=50,
        guidance_scale = 4.0
        strength = 0.99
        max_timestep = 500
    ):
        """
        Initialize the Stable Diffusion inpainting pipeline.

        Args:
        - model_path (str): Path to the .safetensors model file.
        - device (str): Device to use for inference ('cuda' or 'cpu').
        - dtype (torch.dtype): Data type to use for the pipeline (e.g., torch.float16).
        - neg_prompt (str): Default negative prompt to use for inpainting.
        - inpaint_timestep_num (int): Number of timesteps for the inpainting scheduler.
        """
        self.device = device
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            model_path,
            torch_dtype=dtype
        )
        
        # Default negative prompt if none provided
        self.neg_prompt = neg_prompt
        self.inpaint_timestep_num = inpaint_timestep_num
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.max_timestep = max_timestep

        # Set scheduler timesteps
        self.inpaint_pipe.scheduler.config.num_train_timesteps = self.max_timestep
        self.inpaint_pipe.scheduler.set_timesteps(self.inpaint_timestep_num, device)

    def forward(self, image, mask, pos_prompt):
        """
        Perform inpainting on the given image and mask.

        Args:
        - image (PIL.Image): The input image to be inpainted.
        - mask (PIL.Image): The mask indicating regions to inpaint.
        - pos_prompt (str): Positive prompt describing what should be generated.
        
        Returns:
        - PIL.Image: The inpainted image.
        """
        # Move the model to the appropriate device (e.g., CUDA)
        self.inpaint_pipe.to(self.device)

        # Perform the inpainting process
        inpaint_missmatched_result = self.inpaint_pipe(
            prompt=pos_prompt,
            image=image,
            mask_image=mask
            negative_prompt=self.neg_prompt,
            strength=self.strength,
            num_inference_steps=self.inpaint_timestep_num,
            guidance_scale=self.guidance_scale,
        ).images[0]

        # Move the model back to CPU to free up GPU memory
        self.inpaint_pipe.to('cpu')
        torch.cuda.empty_cache()

        return inpaint_missmatched_result