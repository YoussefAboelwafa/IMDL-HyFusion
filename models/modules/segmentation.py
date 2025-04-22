import cv2
import numpy as np
from ultralytics import YOLO
import torch
import warnings
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F

def get_semantic_map(image, yolo_model_path="yolo11x-seg.pt", mask_threshold=0.5, device=None):
    """
    Given a batch of input images and a YOLO segmentation model,
    run inference and produce a batch of semantic maps where each pixel is labeled with its class id.
    
    Args:
        image (Tensor): Batch of input images in (Batch, Channel, Height, Width) format.
        yolo_model_path (str): Path to the YOLO segmentation model.
        mask_threshold (float): Threshold for converting soft masks to binary.
    
    Returns:
        semantic_maps (np.array): Batch of single-channel maps (dtype=np.uint8) where pixel values indicate class labels.
                                  Shape: (Batch, Height, Width). 0 is reserved for background.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    # Load YOLO model
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Suppress warnings more efficiently
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=logging.WARNING)
        
        # Load YOLO model (with caching)
        model = get_cached_yolo_model(yolo_model_path).to(device)
        
        # Process batch efficiently
        semantic_maps = []
        with torch.no_grad():  # Prevent memory leaks
            for img in image:
                # Optimize image preprocessing
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_np = cv2.normalize(img_np, None, 0, 1, cv2.NORM_MINMAX)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                
                # Run inference
                results = model(img_rgb, verbose=False)
                result = results[0]
                
                # Create semantic map efficiently
                semantic_map = torch.zeros((img.shape[1], img.shape[2]), 
                                        dtype=torch.uint8, device=device)
                
                if result.masks is not None:
                    masks = result.masks.data
                    classes = result.boxes.cls.to(device)
                    
                    # Vectorized operations for mask processing
                    for mask, cls in zip(masks, classes):
                        mask = F.interpolate(mask.unsqueeze(0), 
                                          size=(semantic_map.shape[0], semantic_map.shape[1]),
                                          mode='nearest').squeeze(0)
                        semantic_map[mask > mask_threshold] = cls.int() + 1
                
                semantic_maps.append(semantic_map)
        
        return torch.stack(semantic_maps)

def get_cached_yolo_model(model_path):
    """Cache YOLO model to prevent reloading"""
    if not hasattr(get_cached_yolo_model, 'cache'):
        get_cached_yolo_model.cache = {}
    
    if model_path not in get_cached_yolo_model.cache:
        get_cached_yolo_model.cache[model_path] = YOLO(model_path)
    
    return get_cached_yolo_model.cache[model_path]