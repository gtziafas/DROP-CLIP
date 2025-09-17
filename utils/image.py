import cv2
import numpy as np
import torch


def seg_mask_to_binary(seg):
    obj_ids = sorted(np.unique(seg)) # remove background
    return np.stack([seg==obj for obj in obj_ids], axis=0) # (K,H,W)
            

def binary_masks_to_seg(masks, obj_ids=None):
    if obj_ids is None:
        obj_ids = np.arange(masks.shape[0], dtype=np.uint8)
    seg = np.max(masks * obj_ids[:, None, None], axis=0)  # Vectorized operation
    return seg


def seg_continuous_ids(seg):
    unique_ids, new_ids = np.unique(seg, return_inverse=True)
    seg_continuous = new_ids.reshape(seg.shape)
    return seg_continuous.astype(np.uint8)


# get contour out of binary mask
def get_mask_contour(binary_mask):
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8) * 255, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all contours into one array
    #all_contours = np.vstack(contours[i] for i in range(len(contours)))
    #all_contours = np.vstack(contours)
    all_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return all_contours[0]


# get bbox out of binary mask
def get_mask_bbox(binary_mask):
    cont = get_mask_contour(binary_mask)
    x, y, w, h = cv2.boundingRect(cont)
    return [x,y,w,h]


def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1


def add_borders_to_image(image, target_ratio, use_color):
    h, w = image.shape[:2]  # Assuming image is grayscale or color (HxW or HxWx3)
    current_ratio = w / h

    if current_ratio > target_ratio:  # Image is too wide
        new_h = int(w / target_ratio)
        padding = (new_h - h) // 2
        new_image = use_color * np.ones((new_h, w) + image.shape[2:], dtype=image.dtype)
        new_image[padding:padding+h, :] = image
    elif current_ratio < target_ratio:  # Image is too narrow
        new_w = int(h * target_ratio)
        padding = (new_w - w) // 2
        new_image = use_color * np.ones((h, new_w) + image.shape[2:], dtype=image.dtype)
        new_image[:, padding:padding+w] = image
    else:  # Image already has the correct aspect ratio
        new_image = image

    return new_image


def mask2box_multi_level(binary_mask, level, expansion_ratio=0.1):
    # x1, y1, w, h  = get_mask_bbox(binary_mask)
    # x2, y2 = x1 + w, y1 + h
    x1, y1, x2, y2 = mask2box(torch.from_numpy(binary_mask))
    if level == 0:
        return x1, y1, x2, y2
    shape = binary_mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)


# fit ellipse in contour out of binary mask
def get_mask_ellipse(binary_mask):
    cont = get_mask_contour(binary_mask)
    try:
        ellipse = cv2.fitEllipse(cont)
    except:
        x,y,w,h = cv2.boundingRect(cont)
        ellipse = ((x,y), (w,h), 0)
    return ellipse

# get center of mask
def get_mask_center(binary_mask):
    x, y, w, h = get_mask_bbox(binary_mask)
    return int(x + w/2), int(y + h/2)


def get_oriented_bounding_box(binary_mask):
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8) * 255, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all contours into one array
    #all_contours = np.vstack(contours[i] for i in range(len(contours)))
    all_contours = np.vstack(contours)
    
    # Get the oriented bounding box for the combined contours
    rect = cv2.minAreaRect(all_contours)
    
    # Get the box coordinates and cast them to int
    box = np.int0(cv2.boxPoints(rect))
    
    return box