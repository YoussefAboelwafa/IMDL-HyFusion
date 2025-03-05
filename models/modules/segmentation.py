import cv2
import numpy as np
from ultralytics import YOLO

import matplotlib.pyplot as plt
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
    # Load YOLO model
    model = YOLO(yolo_model_path)
    
    # Initialize list to store semantic maps
    semantic_maps = []
    
    # Iterate over each image in the batch
    for img in image:
        # Convert image tensor to numpy array and squeeze batch dimension
        img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to (Height, Width, Channel)
        print(img.shape)
        plt.imshow(img)
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
                semantic_map[binary_mask] = cls + 1
        
        # Append the semantic map to the list
        plt.imshow(semantic_map)
        semantic_maps.append(semantic_map)
    
    # Convert list of semantic maps to numpy array
    semantic_maps = np.array(semantic_maps)
    
    return semantic_maps