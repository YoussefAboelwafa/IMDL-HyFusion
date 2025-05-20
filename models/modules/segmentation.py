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
def get_semantic_map(image, yolo_model_path="yolov8n-seg.pt", mask_threshold=0.5, device=None):
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
    batch_size = image.shape[0]
    target_h, target_w = image.shape[2:4]

    # Create output tensor directly with correct batch size
    semantic_maps = torch.zeros((batch_size, target_h, target_w), 
                               dtype=torch.uint8, 
                               device=device)

    with warnings.catch_warnings(), torch.no_grad():
        # Load YOLO model (with caching)
        model = get_cached_yolo_model(yolo_model_path).to(device)

        # Process images in batches for better efficiency
        # Convert entire batch at once to save processing time
        img_batch = []
        for img in image:
            # Optimize image preprocessing with vectorized operations
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = cv2.normalize(img_np, None, 0, 1, cv2.NORM_MINMAX)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_batch.append(img_rgb)

        # Run inference on batch (YOLO handles batching internally)
        results = model(img_batch, verbose=False)

        # Process results for each image
        for idx, result in enumerate(results):
            if result.masks is not None:
                masks = result.masks.data  # Shape: (num_masks, H, W)
                classes = result.boxes.cls.to(device)

                if len(masks) > 0:
                    # Process all masks at once for better efficiency
                    # Stack all masks into a single tensor
                    stacked_masks = torch.stack([mask for mask in masks])

                    # Add batch and channel dimensions for interpolation
                    stacked_masks = stacked_masks.unsqueeze(1)  # Shape: (num_masks, 1, H, W)

                    # Interpolate all masks at once
                    stacked_masks = F.interpolate(
                        stacked_masks, 
                        size=(target_h, target_w),
                        mode='nearest'
                    )

                    # Process each mask with vectorized operations
                    for i, (mask, cls) in enumerate(zip(stacked_masks, classes)):
                        mask = mask.squeeze()  # Back to (H, W)
                        class_value = (cls.int() + 1).to(torch.uint8)
                        semantic_maps[idx][mask > mask_threshold] = class_value

    return semantic_maps

def get_cached_yolo_model(model_path):
    """Cache YOLO model to prevent reloading"""
    if not hasattr(get_cached_yolo_model, 'cache'):
        get_cached_yolo_model.cache = {}

    if model_path not in get_cached_yolo_model.cache:
        get_cached_yolo_model.cache[model_path] = YOLO(model_path)

    return get_cached_yolo_model.cache[model_path]
