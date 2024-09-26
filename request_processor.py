from PIL import Image, ImageChops

import torch
import numpy as np

from utils_mask import get_mask_location
from torchvision import transforms

from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from segmentation_processor import request_segmentation_results, extract_submask
from vto_core_module import apply_net

def correct_masking(preserve_mask, org_image, mask, mask_gray):
    preserve_mask = Image.fromarray(preserve_mask).convert('L')
    mask2_inverted = ImageChops.invert(preserve_mask)
    corrected_mask = ImageChops.multiply(mask, mask2_inverted)
    corrected_mask_gray = Image.composite(org_image, mask_gray, preserve_mask)
    return corrected_mask, corrected_mask_gray
    

class TryOnProcessor:
    def __init__(self, pipeline_config, pipeline_loader):
        self.pipeline_config = pipeline_config
        self.segmentaion_config = self.pipeline_config['segmentaion']
        self.device = pipeline_config['device']
        self.vto_core = pipeline_loader.get_vto_core()
        self.openpose_model = pipeline_loader.get_openpose_model()
        self.parsing_model = pipeline_loader.get_parsing_model()
        self.inpainting_diffusion = pipeline_loader.get_inpainting_diffusion()
        self.attribute_classifier = pipeline_loader.get_attributes_classifier()

    def to(self, device):
        self.vto_core.to(device)

    def preprocess_submasks(self, init_image):
        init_segmentation_map, init_classes_mapping = request_segmentation_results(
            url=self.segmentaion_config['service_url'], 
            image=init_image
        )

        pre_preservation_classes = extract_submask(
            segmentation_map=init_segmentation_map,
            submask_classes=self.segmentaion_config['pre_preservation_classes'],
            classes_mapping=init_classes_mapping
        )

        return pre_preservation_classes, init_segmentation_map, init_classes_mapping
        

    def postprocess_submasks(
            self, 
            init_image, 
            init_segmentation_map,
            init_classes_mapping, 
            result_image):
        segmentaion_config = self.segmentaion_config

        soft_preservation_submask = extract_submask(
            segmentation_map=init_segmentation_map,
            submask_classes=segmentaion_config['soft_preservation_classes'],
            classes_mapping=init_classes_mapping
        )

        force_preservetion_submask = extract_submask(
            segmentation_map=init_segmentation_map,
            submask_classes=segmentaion_config['force_preservation_classes'],
            classes_mapping=init_classes_mapping
        )

        result_segmentation_map, result_classes_mapping = request_segmentation_results(
            url=segmentaion_config['service_url'], 
            image=result_image
        )

        clothing_submask = extract_submask(
            segmentation_map=result_segmentation_map,
            submask_classes=segmentaion_config['clothing_classes'],
            classes_mapping=result_classes_mapping
        )

        Image.fromarray(clothing_submask).convert("L").save('clothing_submask.png')

        force_mask = Image.fromarray(force_preservetion_submask).convert("L")
        composed_image = Image.composite(init_image, result_image, force_mask)

        soft_mask = np.logical_and(
            soft_preservation_submask, 
            np.logical_not(clothing_submask)
        )
        soft_mask_pil = Image.fromarray(soft_mask).convert("L")

        composed_image = Image.composite(init_image, composed_image, soft_mask_pil)
        return composed_image


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
                './vto_core_module/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                './vto_core_module/ckpt/densepose/model_final_162be9.pkl',
                'dp_segm', '-v', '--opts', 'MODEL.DEVICE', self.device
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        return pose_img

    def start_tryon(
        self, human_canva, garm_img, garment_des, denoise_steps, seed
    ):
        # Preprocess images
        garm_img, human_img, human_img_orig = self.preprocess_images(human_canva, garm_img)
        org_size = human_img_orig.size

        (
            pre_preservation_classes,
            init_segmentation_map,
            init_classes_mapping
        ) = self.preprocess_submasks(init_image=human_img)

        # Generate keypoints and parse model
        keypoints, model_parse = self.generate_keypoints_and_parse_model(human_img)

        # Generate mask and mask_gray
        mask, mask_gray = self.generate_mask_and_mask_gray(model_parse, keypoints, human_img)

        mask, mask_gray = correct_masking(
            preserve_mask=pre_preservation_classes, 
            org_image=human_img,
            mask=mask,
            mask_gray=mask_gray
        )

        # Prepare human image for pose estimation
        human_img_arg = self.prepare_human_image_for_pose_estimation(human_img)

        # Generate pose image
        pose_img = self.generate_pose_image(human_img_arg)

        result_image = self.vto_core.process_vto_pipeline(
            pose_img,
            garm_img,
            human_img,
            garment_des,
            mask,
            denoise_steps,
            seed
        )

        compose_result = self.postprocess_submasks(
            init_image=human_img,
            init_segmentation_map=init_segmentation_map,
            init_classes_mapping=init_classes_mapping,
            result_image=result_image,
        )
        compose_result = compose_result.resize(org_size)

        return compose_result, mask_gray
