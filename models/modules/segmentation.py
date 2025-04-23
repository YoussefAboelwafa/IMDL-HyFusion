import cv2
import numpy as np
from ultralytics import YOLO
import torch
import warnings
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F
warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
warnings.filterwarnings("ignore", message="libpng warning: iCCP: profile 'ICC profile': 'bTRC': ICC profile tag start not a multiple of 4")
warnings.filterwarnings("ignore", message="Corrupt EXIF data", module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
def get_semantic_map(image, yolo_model_path="yolo11x-seg.pt", mask_threshold=0.5, device=None):
    """
    Given a batch of input images and a YOLO segmentation model,
    run inference and produce a batch of semantic maps where each pixel is labeled with its class id.
    
    Args:
        image (Tensor): Batch of input images in (Batch, Channel, Height, Width) format.
        yolo_model_path (str): Path to the YOLO segmentation model.
        mask_threshold (float): Threshold for converting soft masks to binary.
    
    Returns:
        semantic_maps (torch.Tensor): Batch of single-channel maps where pixel values indicate class labels.
                                    Shape: (Batch, Height, Width)
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    # Set device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process batch efficiently
    semantic_maps = []
    with warnings.catch_warnings(), torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        # Load YOLO model (with caching)
        model = get_cached_yolo_model(yolo_model_path).to(device)
        
        for img in image:
            # Optimize image preprocessing
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = cv2.normalize(img_np, None, 0, 1, cv2.NORM_MINMAX)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Get target size for interpolation
            target_h, target_w = img.shape[1:3]
            
            # Create semantic map efficiently
            semantic_map = torch.zeros((target_h, target_w), dtype=torch.uint8, device=device)
            
            # Run inference
            results = model(img_rgb, verbose=False)
            result = results[0]
            
            if result.masks is not None:
                masks = result.masks.data  # Shape: (num_masks, H, W)
                classes = result.boxes.cls.to(device)
                
                # Process each mask
                for mask, cls in zip(masks, classes):
                    # Ensure mask has correct dimensions (1, 1, H, W) for interpolation
                    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                    
                    # Interpolate mask to target size
                    mask = F.interpolate(mask, 
                                      size=(target_h, target_w),
                                      mode='nearest')
                    
                    # Remove extra dimensions and apply threshold
                    mask = mask.squeeze()  # Back to (H, W)
                    class_value = (cls.int() + 1).to(torch.uint8)
                    semantic_map[mask > mask_threshold] = class_value
            
            semantic_maps.append(semantic_map)
    
    return torch.stack(semantic_maps)

def get_cached_yolo_model(model_path):
    """Cache YOLO model to prevent reloading"""
    if not hasattr(get_cached_yolo_model, 'cache'):
        get_cached_yolo_model.cache = {}
    
    if model_path not in get_cached_yolo_model.cache:
        get_cached_yolo_model.cache[model_path] = YOLO(model_path)
    
    return get_cached_yolo_model.cache[model_path]
