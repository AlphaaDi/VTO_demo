import cv2
import torch
import numpy as np
import requests
from io import BytesIO
import yaml
from PIL import Image, ImageChops, ImageFilter

from utils_mask import get_mask_location, erode_mask, remove_small_clusters_np, erode_based_on_distance
from torchvision import transforms
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from segmentation_processor import request_segmentation_results, extract_submask, get_all_submasks, join_submasks
import vto_core_module.apply_net as apply_net
from hands_mask_extender import expand_arms_compose_masking


def pil_image_to_bytes(pil_image, format='PNG'):
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

def correct_masking(preserve_mask, org_image, mask, mask_gray):
    preserve_mask = Image.fromarray(preserve_mask).convert('L')
    mask2_inverted = ImageChops.invert(preserve_mask)
    corrected_mask = ImageChops.multiply(mask, mask2_inverted)
    corrected_mask_gray = Image.composite(org_image, mask_gray, preserve_mask)
    return corrected_mask, corrected_mask_gray


def add_gaussian_noise(image, mean=30, stddev=10):
    # Convert PIL Image to NumPy array
    img_array = np.array(image).astype(int)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, img_array.shape).astype(int)

    # Add the noise to the image
    noisy_image_array = img_array + noise

    # Clip the values to be in a valid range (0, 255)
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert the NumPy array back to a PIL Image
    noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))

    return noisy_image


def request_inpainting(
    image, 
    mask, 
    ip_adapter_image, 
    pos_prompt,
    sd_inpainting_config,
):
    # Convert PIL Images to BytesIO streams
    image_stream = pil_image_to_bytes(image)
    mask_stream = pil_image_to_bytes(mask)
    ip_adapter_image_stream = pil_image_to_bytes(ip_adapter_image)
    
    # Prepare the files dictionary
    files = {
        'image': ('image.png', image_stream, 'image/png'),
        'mask': ('mask.png', mask_stream, 'image/png'),
        'ip_adapter_image': ('ip_adapter_image.png', ip_adapter_image_stream, 'image/png'),
    }
    
    # Prepare the form data
    data = {
        'pos_prompt': pos_prompt,
    }

    server_config = sd_inpainting_config['server']
    port = server_config['port']
    host = server_config['host']
    method_name = server_config['method_name']
    
    inpainting_url = f'http://{host}:{port}/{method_name}'
    # Send the POST request to the API
    response = requests.post(inpainting_url, files=files, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        raise BrokenPipeError(
            f'Error in stable diffusion inpainting server: {response.status_code}: {response.text}'
        )
    return image
    

class TryOnProcessor:
    def __init__(self, pipeline_config, pipeline_loader):
        self.pipeline_config = pipeline_config
        self.segmentaion_config = self.pipeline_config['segmentaion']
        self.device = pipeline_config['device']
        self.vto_core = pipeline_loader.get_vto_core()
        self.openpose_model = pipeline_loader.get_openpose_model()
        self.parsing_model = pipeline_loader.get_parsing_model()
        self.attribute_classifier = pipeline_loader.get_attributes_classifier()
        self.garment_description_generator = pipeline_loader.get_garment_description_generator()

        self.tensor_transform = self.vto_core.get_tensor_transform()

        with open(pipeline_config['sd_inpainting_config_path'], 'r') as file:
            self.sd_inpainting_config = yaml.safe_load(file)


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

        pre_preservation_classes = erode_mask(pre_preservation_classes)

        return pre_preservation_classes, init_segmentation_map, init_classes_mapping
        

    def get_arms_hands_split_masks(self, submasks):
        arms_class_names = self.segmentaion_config['soft_preservation_classes']
        hands_cls = self.segmentaion_config['hands_classes']
        arms_wo_hands_cls = set(arms_class_names) - set(hands_cls)
        
        arms_wo_hands_submask = join_submasks(
            submasks, arms_wo_hands_cls
        )

        hands_submask = join_submasks(
            submasks, hands_cls
        )
        return arms_wo_hands_submask, hands_submask


    def postprocess_submasks(
        self, 
        init_image, 
        init_submasks,
        result_submasks,
        result_image,
        erode_size = 7,
        erosion_size_final=3,
    ):
        arms_wo_hands_init_submask, hands_init_submask = self.get_arms_hands_split_masks(init_submasks)

        upper_clothe_mask = join_submasks(
            init_submasks, ['Upper_Clothing']
        )

        arms_wo_hands_init_submask = erode_based_on_distance(
            arms_wo_hands_init_submask, 
            upper_clothe_mask, 
            threshold = erode_size,
            kernel_size=erode_size,
        )

        soft_preservation_submask = np.logical_or(arms_wo_hands_init_submask, hands_init_submask)

        
        clothing_submask = join_submasks(
            result_submasks, self.segmentaion_config['clothing_classes']
        )

        soft_mask = np.logical_and(
            soft_preservation_submask, 
            np.logical_not(clothing_submask)
        )
        soft_mask = remove_small_clusters_np(soft_mask, min_size=1000)
        soft_mask_pil = Image.fromarray(soft_mask).convert("L")
        soft_mask_pil = erode_mask(soft_mask_pil,erosion_size=erosion_size_final)

        composed_image = Image.composite(init_image, result_image, soft_mask_pil)

        return composed_image
    

    def get_union_mask(self, init_submasks, result_submasks, blur_size=5):
        arms_wo_hands_init_submask, hands_init_submask = self.get_arms_hands_split_masks(init_submasks)
        
        arms_result_submask =  join_submasks(
            result_submasks, self.segmentaion_config['hands_classes']
        )
        
        union_mask = np.logical_or(arms_wo_hands_init_submask, arms_result_submask)
        union_mask = np.logical_and(union_mask, np.logical_not(hands_init_submask))
        union_mask = cv2.dilate(
            union_mask.astype(np.uint8), 
            kernel=np.ones((3, 3), np.uint8),
            iterations=1
        ) 
        union_mask = union_mask > 0
        
        union_mask_pil = Image.fromarray(union_mask).convert('L')
        union_mask_pil = union_mask_pil.filter(ImageFilter.GaussianBlur(blur_size))
        return union_mask_pil


    def get_more_compose_result(self, human_img, init_submasks, result_submasks, keypoints_res, compose_result):
        more_compose_masks = expand_arms_compose_masking(
            human_img,
            init_submasks,
            result_submasks,
            keypoints_res,
            erode_size=10,
        )

        if more_compose_masks is None:
            return compose_result
        
        more_compose_masks_pil = Image.fromarray(more_compose_masks)
        more_compose_masks_pil = more_compose_masks_pil.convert('L')
        more_compose_result = Image.composite(human_img, compose_result, more_compose_masks_pil)
    
        return more_compose_result

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

    def process_tryon(
        self, human_canva, garm_img, garment_des=None, denoise_steps=50, seed=997
    ):
        if garment_des is None:
            garment_des = self.garment_description_generator.get_description(garm_img)
        # Preprocess images
        garm_img, human_img, human_img_orig = self.preprocess_images(human_canva, garm_img)
        org_size = human_img_orig.size

        attributes = self.attribute_classifier.forward(human_img)
        
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
            seed,
            attributes=attributes
        )
        
        
        result_image_noised = add_gaussian_noise(result_image, mean=30, stddev=10)
        result_segmentation_map, result_classes_mapping = request_segmentation_results(
            url=self.segmentaion_config['service_url'], 
            image=result_image_noised
        )
        
        init_submasks = get_all_submasks(
            segmentation_map=init_segmentation_map,
            classes_mapping=init_classes_mapping
        )
        
        result_submasks = get_all_submasks(
            segmentation_map=result_segmentation_map,
            classes_mapping=result_classes_mapping
        )

        compose_result = self.postprocess_submasks(
            init_image=human_img,
            init_submasks=init_submasks,
            result_submasks=result_submasks,
            result_image=result_image,
        )
        
        keypoints_res, _ = self.generate_keypoints_and_parse_model(result_image)
        
        more_compose_result = self.get_more_compose_result(
            human_img, 
            init_submasks, 
            result_submasks,
            keypoints_res, 
            compose_result
        )
        
        union_mask_pil = self.get_union_mask(
            init_submasks,
            result_submasks
        )

        attributes['garment_des'] = garment_des
        positive_prompt = self.pipeline_config["core_config"]["prompt"].format(**attributes)
        
        inpainting_result = request_inpainting(
            image=more_compose_result,
            mask=union_mask_pil,
            ip_adapter_image=human_img,
            pos_prompt=positive_prompt,
            sd_inpainting_config=self.sd_inpainting_config
        )

        inpainting_result_res = inpainting_result.resize(org_size)
        result_image_res = result_image.resize(org_size)
        
        return inpainting_result_res
