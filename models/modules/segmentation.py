import cv2
import numpy as np
from ultralytics import YOLO

def get_semantic_map(image, yolo_model_path="yolov8n-seg.pt", mask_threshold=0.5):
    """
    Given an input image and a YOLO segmentation model,
    run inference and produce a semantic map where each pixel is labeled with its class id.
    
    Args:
        image (np.array): Input image in BGR format.
        model (YOLO): YOLO segmentation model.
        mask_threshold (float): Threshold for converting soft masks to binary.
    
    Returns:
        semantic_map (np.array): A single-channel map (dtype=np.uint8) where pixel values indicate class labels.
                                 0 is reserved for background.
    """
    # Convert image from BGR to RGB since YOLO expects RGB
    model = YOLO(yolo_model_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference on the image
    results = model(image_rgb)
    result = results[0]  # Assumes one image
    
    # Initialize semantic map with zeros (background label)
    semantic_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if result.masks is not None:
        # Extract instance masks and corresponding class labels
        masks = result.masks.data.cuda().numpy()  # shape: [num_instances, height, width]
        classes = result.boxes.cls.cuda().numpy().astype(np.int32)
        
        # Process each instance mask: threshold and assign class id (offset by 1 so that 0 remains background)
        for mask, cls in zip(masks, classes):
            binary_mask = mask > mask_threshold
            semantic_map[binary_mask] = cls + 1
            
    return semantic_map
