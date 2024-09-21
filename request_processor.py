from PIL import Image

import torch
import numpy as np
import json

from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from gradio_client import Client, handle_file

from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image


def request_segmentation_results(url="http://127.0.0.1:7860/"):
    client = Client(url)
    _, npy_path, json_path = client.predict(
            image=handle_file('example/human/Screenshot 2024-09-20 at 20.21.32.png'),
            model_name="1b",
            api_name="/process_image"
    )
    class_segmentation = np.load(npy_path)

    with open(json_path, 'r') as f:
        class_mapping = json.load(f)

    return class_segmentation, class_mapping

def create_submask(segmentation_map, submask_classes, class_mapping):
    """
    Creates a submask of the union of the classes to preserve.

    Args:
    segmentation_map (numpy.ndarray): A 2D array of argmax indices representing the segmentation map.
    submask_classes (list): List of class.
    class_mapping (dict): A dictionary mapping class names to their corresponding index values.

    Returns:
    numpy.ndarray: A binary submask where the classes to preserve are set to 1 and the rest are 0.
    """
    # Convert the class names to indices using the class mapping
    class_indices_to_preserve = [class_mapping[cls] for cls in submask_classes]

    # Create a binary mask where classes to preserve are 1, and others are 0
    submask = np.isin(segmentation_map, class_indices_to_preserve).astype(np.uint8)

    return submask


class TryOnProcessor:
    def __init__(self, device, pipe, openpose_model, parsing_model, tensor_transform):
        self.device = device
        self.pipe = pipe
        self.openpose_model = openpose_model
        self.parsing_model = parsing_model
        self.tensor_transform = tensor_transform

    def preprocess_images(self, human_canva, garm_img):
        garm_img = garm_img.convert("RGB").resize((768, 1024))
        human_img_orig = human_canva["background"].convert("RGB")
        human_img = human_img_orig.resize((768, 1024))
        return garm_img, human_img, human_img_orig

    def generate_keypoints_and_parse_model(self, human_img):
        resized_human_img = human_img.resize((384, 512))
        keypoints = self.openpose_model(resized_human_img)
        model_parse, _ = self.parsing_model(resized_human_img)
        return keypoints, model_parse

    def generate_mask_and_mask_gray(self, model_parse, keypoints, human_img):
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        return mask, mask_gray

    def prepare_human_image_for_pose_estimation(self, human_img):
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        return human_img_arg

    def generate_pose_image(self, human_img_arg):
        argument_parser = apply_net.create_argument_parser()
        args = argument_parser.parse_args(
            (
                'show',
                './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                './ckpt/densepose/model_final_162be9.pkl',
                'dp_segm', '-v', '--opts', 'MODEL.DEVICE', self.device
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        return pose_img

    def encode_prompts(self, garment_des):
        prompt = "model is wearing " + garment_des
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
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

        prompt = "a photo of " + garment_des
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

    def generate_images_with_model(
        self,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        denoise_steps,
        generator,
        pose_img_tensor,
        prompt_embeds_c,
        garm_tensor,
        mask,
        human_img,
        garm_img,
    ):
        images = self.pipe(
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
        )[0]
        return images

    def start_tryon(
        self, human_canva, garm_img, garment_des, denoise_steps, seed
    ):
        # Preprocess images
        garm_img, human_img, human_img_orig = self.preprocess_images(human_canva, garm_img)

        # Generate keypoints and parse model
        keypoints, model_parse = self.generate_keypoints_and_parse_model(human_img)

        # Generate mask and mask_gray
        mask, mask_gray = self.generate_mask_and_mask_gray(model_parse, keypoints, human_img)

        # Prepare human image for pose estimation
        human_img_arg = self.prepare_human_image_for_pose_estimation(human_img)

        # Generate pose image
        pose_img = self.generate_pose_image(human_img_arg)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Encode prompts
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    prompt_embeds_c,
                ) = self.encode_prompts(garment_des)

                # Prepare images for the model
                pose_img_tensor, garm_tensor = self.prepare_images_for_model(pose_img, garm_img)

                generator = (
                    torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                )

                # Generate images with the model
                images = self.generate_images_with_model(
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    denoise_steps,
                    generator,
                    pose_img_tensor,
                    prompt_embeds_c,
                    garm_tensor,
                    mask,
                    human_img,
                    garm_img,
                )

        return images[0], mask_gray
