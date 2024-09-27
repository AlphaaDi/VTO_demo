import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]
        

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


def erode_mask(org_mask, erosion_size=3):
    '''
        - mask (np.ndarray): Binary mask with 0s and 255s.
    '''
    if isinstance(org_mask, Image.Image):
        org_mask = org_mask.convert('L')

    mask = np.array(org_mask)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask, erosion_kernel, iterations=1)
    _, final_mask = cv2.threshold(eroded_mask, 127, 255, cv2.THRESH_BINARY)
    
    if isinstance(org_mask, Image.Image):
        final_mask = Image.fromarray(final_mask).convert('L')
    return final_mask


def fill_holes(org_mask, min_size):
    """
    Removes small isolated clusters from a binary mask.
    
    :param binary_mask: Binary mask (numpy array)
    :param min_size: Minimum size of clusters to keep
    :return: Cleaned binary mask with small clusters removed
    """

    if isinstance(org_mask, Image.Image):
        org_mask = org_mask.convert('L')

    binary_mask = np.array(org_mask) > 0
    binary_mask = binary_mask.astype(np.uint8)

    # Find all connected components (clusters)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Initialize an output mask to store the result
    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    
    # Iterate through all the components and filter based on size
    for i in range(1, num_labels):  # Start from 1 to skip the background
        component_size = stats[i, cv2.CC_STAT_AREA]
        if component_size >= min_size:
            # Keep the component if it's larger than the minimum size
            cleaned_mask[labels == i] = 255

    cleaned_mask = cleaned_mask.astype(bool)

    if isinstance(org_mask, Image.Image):
        cleaned_mask = Image.fromarray(cleaned_mask).convert('L')

    return cleaned_mask


def remove_small_clusters_np(org_mask, min_size):
    cleaned_mask = fill_holes(org_mask, min_size)
    cleaned_mask_inv = ~cleaned_mask
    cleaned_mask_inv_v2 = fill_holes(cleaned_mask_inv, min_size)
    cleaned_mask = ~cleaned_mask_inv_v2
    return cleaned_mask

def filter_points_by_distance(reference_point, points_list, max_distance):
    filtered_points = []

    # Unpack the reference point
    x_ref, y_ref = reference_point

    # Iterate over the points list and calculate the distance
    for x, y in points_list:
        distance = ((x - x_ref) ** 2 + (y - y_ref) ** 2) ** (1/2)
        if distance <= max_distance:
            filtered_points.append((x, y))
    
    return filtered_points


def draw_binary_mask_on_image(base_image, binary_mask, color=(0, 255, 0)):
    """
    Composes a binary mask on top of a PIL base image with a specific color (default green).

    Args:
    - base_image (PIL.Image): The background image.
    - binary_mask (np.ndarray or PIL.Image): Binary mask image (0 and 255 values).
    - color (tuple): Color to apply to the mask (R, G, B). Default is green (0, 255, 0).

    Returns:
    - PIL.Image: The resulting image with the mask composed on top.
    """
    # Ensure base_image is in RGB mode
    base_image = base_image.convert('RGB')

    # Convert the binary_mask to a NumPy array if it's a PIL Image
    if isinstance(binary_mask, Image.Image):
        binary_mask = np.array(binary_mask)

    # Create a blank image with the same size as the base image, filled with the color
    color_mask = np.zeros_like(base_image)
    color_mask[..., 0] = color[0]
    color_mask[..., 1] = color[1]
    color_mask[..., 2] = color[2]

    # Ensure the binary mask is 0 and 255 (binary format)
    binary_mask = np.where(binary_mask > 0, 127, 0).astype(np.uint8)

    # Use the binary mask to composite the green color mask over the base image
    mask_image = Image.fromarray(binary_mask).convert("L")  # Convert mask to grayscale
    color_mask_image = Image.fromarray(color_mask)  # Convert color mask back to PIL

    # Composite the color mask over the base image where the binary mask is true
    composed_image = Image.composite(color_mask_image, base_image, mask_image)

    return composed_image

def erode_based_on_distance(mask1, mask2, threshold, kernel_size=3):
    # Compute the distance transform of the second mask (distance to nearest non-zero pixel)
    distance_map = distance_transform_edt(1 - mask2)
    
    # Create a mask of pixels where the distance is below the threshold
    close_to_mask2 = distance_map < threshold
    
    # Create a kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    mask1 = mask1.astype(np.uint8)
    # Erode the first mask
    eroded_mask1 = cv2.erode(mask1, kernel)
    
    # Apply erosion only in regions where distance to mask2 is less than the threshold
    final_mask = np.where(close_to_mask2, eroded_mask1, mask1)
    
    return final_mask