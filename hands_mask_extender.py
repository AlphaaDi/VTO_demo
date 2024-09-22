import cv2
import numpy as np
from utils_mask import erode_mask, filter_points_by_distance

def resize_keypoints_back(keypoints, img_resized_size, img_org_size):
    resized_width, resized_height = img_resized_size
    original_width, original_height = img_org_size

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    resized_keypoints = []
    for point in keypoints:
        x, y = point
        if x == 0 and y == 0:
            resized_keypoints.append([0, 0])
        else:
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            resized_keypoints.append([new_x, new_y])

    return resized_keypoints

def _bresenham_line(x1, y1, x2, y2, sizes):
    """
    Standard Bresenham's algorithm for generating points between (x1, y1) and (x2, y2).
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == 0 or x1 == sizes[0]-1 or y1 == 0 or y1 == sizes[1] - 1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def bresenham_line(x1, y1, x2, y2, sizes):
    line_points1 = _bresenham_line(x1, y1, x2, y2, sizes=sizes)
    line_points2 = _bresenham_line(x2, y2, x1, y1, sizes=sizes)
    line_points = list(set(line_points1 + line_points2))
    return line_points

def perpendicular_line(x, y, slope, length, sizes):
    """
    Build a perpendicular line to a given slope, centered at (x, y).
    
    Args:
    - x, y: Point where the perpendicular line originates.
    - slope: Slope of the original line.
    - length: Length of the perpendicular line to generate.
    
    Returns:
    - List of points [(x, y), ...] on the perpendicular line.
    """
    if slope == 0:  # Handle the case where the line is horizontal
        perp_slope = np.inf
    else:
        perp_slope = -1 / slope

    for shift in range(-length//2, length//2 + 1):
        if perp_slope == np.inf:  # Perpendicular to horizontal line
            new_x, new_y = x, y + shift
        else:
            new_x = x + shift
            new_y = int(np.round(y + perp_slope * shift))
        if new_x <= 0 or new_x >= sizes[0]-1 or new_y <= 0 or new_y >= sizes[1] - 1:
            continue
        break
        
    line_points = bresenham_line(x, y, new_x, new_y, sizes=sizes)

    line_points = filter_points_by_distance(
        reference_point=(x, y), 
        points_list=line_points,
        max_distance=length
    )
    
    return line_points

def find_intersections(mask2, points):
    """
    Find the intersection points between the second mask and the points on the perpendicular line.
    
    Args:
    - mask2 (np.ndarray): Second mask where we are looking for intersections.
    - points (list): List of points on the perpendicular line.
    
    Returns:
    - List of points where the line intersects the second mask.
    """
    intersections = []
    for (x, y) in points:
        if 0 <= x < mask2.shape[1] and 0 <= y < mask2.shape[0]:  # Check if within bounds
            if mask2[y, x] > 0:  # Intersection with the second mask
                intersections.append((x, y))
    return intersections

def process_masks(center1, center2, mask1, mask2, perp_length=100):
    """
    Draw a line between the centers of two masks and for each pixel in the first mask along that line,
    find perpendicular lines and their intersections with the second mask.

    Args:
    - mask1 (np.ndarray): First binary mask.
    - mask2 (np.ndarray): Second binary mask.
    - center1 (tuple): Center of the first mask (x1, y1).
    - center2 (tuple): Center of the second mask (x2, y2).
    - perp_length (int): Length of the perpendicular lines to generate.

    Returns:
    - intersections: List of intersection points with mask2.
    """
    x1, y1 = center1
    x2, y2 = center2

    sizes = mask1.shape[::-1]
    # Step 1: Get the points along the line between the two centers
    line_points = bresenham_line(x1, y1, x2, y2, sizes)
                    
    # Step 2: Get the slope of the line between the centers
    slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0  # Handle vertical line case

    intersections = []
    # Step 3: For each point on the line, generate a perpendicular line
    for (x, y) in line_points:
        if mask1[y, x] > 0:  # Only process if the pixel is part of the first mask
            perp_points = perpendicular_line(x, y, slope, perp_length, sizes)
            intersect_points = find_intersections(mask2, perp_points)
            intersections.extend(intersect_points)
    
    return intersections

def get_additional_compose_mask(
        center1,
        center2,
        mask1,
        mask2,
        dilate_size = 7,
        erosion_size = 7,
):
    if center1 is None or center2 is None:
        return None
    if mask1.sum() < 200 or mask2.sum() < 200:
        return None

    intersections = process_masks(
        center1 = center1,
        center2 = center2,
        mask1 = mask1,
        mask2 = mask2
    )
    
    intersection_mask = np.zeros_like(mask1)
    intersections_np = np.array(intersections)
    intersection_mask[intersections_np[:,1], intersections_np[:,0]] = True
    
    dilated_mask = cv2.dilate(
        intersection_mask.astype(np.uint8), 
        kernel=np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    dilated_mask = dilated_mask.astype(bool)
    
    normalized_mask = erode_mask(dilated_mask, erosion_size=erosion_size) 
    normalized_mask = normalized_mask.astype(bool)
    
    return normalized_mask

def expand_arms_compose_masking(
    human_img,
    init_submasks,
    result_submasks,
    keypoints_result,
):
    
    img_resized_size = (384, 512)
    keypoints_resized = resize_keypoints_back(keypoints_result['pose_keypoints_2d'], img_resized_size, human_img.size)
    
    points_idxses = {
        'Right_Shoulder' : 2,
        'Right_Elbo' : 3,
        'Right_Hand' : 4,
        'Left_Shoulder': 5,
        'Left_Elbo' : 6,
        'Left_Hand' : 7,
    }
    
    keypoints_coords = {}
    for name, idx in points_idxses.items():
        keypoints_coords[name] = keypoints_resized[idx]
    
    additional_compose_masks = [
        {
            'mask1': result_submasks['Left_Lower_Arm'],
            'mask2': init_submasks['Left_Lower_Arm'] + init_submasks['Apparel'],
            'center1': keypoints_coords['Left_Elbo'],
            'center2': keypoints_coords['Left_Hand'],
        },
        {
            'mask1': result_submasks['Right_Lower_Arm'],
            'mask2': init_submasks['Right_Lower_Arm'] + init_submasks['Apparel'],
            'center1': keypoints_coords['Right_Elbo'],
            'center2': keypoints_coords['Right_Hand'],
        },
        {
            'mask1': result_submasks['Left_Upper_Arm'],
            'mask2': init_submasks['Left_Upper_Arm'],
            'center1': keypoints_coords['Left_Shoulder'],
            'center2': keypoints_coords['Left_Elbo'],
        },
        {
            'mask1': result_submasks['Right_Upper_Arm'],
            'mask2': init_submasks['Right_Upper_Arm'],
            'center1': keypoints_coords['Right_Shoulder'],
            'center2': keypoints_coords['Right_Elbo'],
        }
    ]
    
    compose_masks = []
    for additional_compose_mask_args in additional_compose_masks:
        compose_mask = get_additional_compose_mask(**additional_compose_mask_args)
        if compose_mask is None:
            continue
        compose_masks.append(compose_mask)
    
    compose_masks = np.array(compose_masks).sum(0)
    compose_masks = compose_masks.astype(bool)
    return compose_masks
