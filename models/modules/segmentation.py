import cv2
import numpy as np
from ultralytics import YOLO
import torch
import warnings
import matplotlib.pyplot as plt
import logging

def get_semantic_map(image, yolo_model_path="yolov8n-seg.pt", mask_threshold=0.5):
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
    model = YOLO(yolo_model_path)
    
    # Initialize list to store semantic maps
    semantic_maps = []
    
    # Iterate over each image in the batch
    for img in image:
        # Convert image tensor to numpy array and squeeze batch dimension
        img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to (Height, Width, Channel)
        
        # Scale image data to the valid range for imshow
        img = (img - img.min()) / (img.max() - img.min())
        
        
        # Convert img from BGR to RGB since YOLO expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference on the img
        results = model(img_rgb)
        result = results[0]  # Assumes one img
        
        # Initialize semantic map with zeros (background label)
        semantic_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        if result.masks is not None:
            # Extract instance masks and corresponding class labels
            masks = result.masks.data.cpu().numpy()  # shape: [num_instances, height, width]
            classes = result.boxes.cls.cpu().numpy().astype(np.int32)
            
            # Process each instance mask: threshold and assign class id (offset by 1 so that 0 remains background)
            for mask, cls in zip(masks, classes):
                binary_mask = mask > mask_threshold
                # Resize binary_mask to match the dimensions of semantic_map
                binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (semantic_map.shape[1], semantic_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                semantic_map[binary_mask_resized.astype(bool)] = cls + 1
        
        # Append the semantic map to the list
        plt.imshow(semantic_map)
        semantic_maps.append(semantic_map)
    
    # Convert list of semantic maps to numpy array
    semantic_maps = np.array(semantic_maps)
    
    return torch.tensor(semantic_maps).to(image.device)