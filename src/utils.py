"""
Utility functions for text segmentation and transliteration
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Union
import json
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file with error handling
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file
    
    Args:
        image: Image array to save
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return cv2.imwrite(output_path, image)
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_box_format(box: Union[List, Tuple], 
                      from_format: str = 'xyxy', 
                      to_format: str = 'xywh') -> List[float]:
    """
    Convert between different bounding box formats
    
    Args:
        box: Bounding box coordinates
        from_format: Source format ('xyxy', 'xywh', 'cxcywh')
        to_format: Target format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        Converted box coordinates
    """
    if from_format == to_format:
        return list(box)
    
    if from_format == 'xyxy':
        x1, y1, x2, y2 = box
        if to_format == 'xywh':
            return [x1, y1, x2 - x1, y2 - y1]
        elif to_format == 'cxcywh':
            return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    
    elif from_format == 'xywh':
        x, y, w, h = box
        if to_format == 'xyxy':
            return [x, y, x + w, y + h]
        elif to_format == 'cxcywh':
            return [x + w / 2, y + h / 2, w, h]
    
    elif from_format == 'cxcywh':
        cx, cy, w, h = box
        if to_format == 'xyxy':
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        elif to_format == 'xywh':
            return [cx - w / 2, cy - h / 2, w, h]
    
    raise ValueError(f"Unknown format conversion: {from_format} to {to_format}")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: First box in xyxy format
        box2: Second box in xyxy format
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_overlapping_boxes(boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Merge overlapping bounding boxes
    
    Args:
        boxes: List of box dictionaries with 'box' and 'score' keys
        iou_threshold: IoU threshold for merging
        
    Returns:
        List of merged boxes
    """
    if not boxes:
        return []
    
    # Sort by score
    boxes = sorted(boxes, key=lambda x: x.get('score', 0), reverse=True)
    merged = []
    
    while boxes:
        current = boxes.pop(0)
        to_merge = [current]
        
        # Find all boxes that overlap with current
        remaining = []
        for box in boxes:
            if calculate_iou(current['box'], box['box']) > iou_threshold:
                to_merge.append(box)
            else:
                remaining.append(box)
        
        # Merge boxes
        if len(to_merge) > 1:
            # Calculate weighted average box
            total_score = sum(b['score'] for b in to_merge)
            merged_box = [0, 0, 0, 0]
            
            for b in to_merge:
                weight = b['score'] / total_score
                for i in range(4):
                    merged_box[i] += b['box'][i] * weight
            
            merged.append({
                'box': merged_box,
                'score': total_score / len(to_merge)
            })
        else:
            merged.append(current)
        
        boxes = remaining
    
    return merged


def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by given angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image and rotation matrix
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    
    # Adjust rotation matrix for translation
    M[0, 2] += (new_width - width) / 2
    M[1, 2] += (new_height - height) / 2
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (new_width, new_height),
                            borderValue=(255, 255, 255))
    
    return rotated, M


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better text detection
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Convert back to BGR if original was color
    if len(image.shape) == 3:
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return sharpened


def save_results_json(results: Dict, output_path: str):
    """
    Save processing results to JSON file
    
    Args:
        results: Results dictionary
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def load_results_json(json_path: str) -> Dict:
    """
    Load processing results from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Results dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_color_map(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Create distinct colors for visualization
    
    Args:
        num_classes: Number of distinct colors needed
        
    Returns:
        List of BGR color tuples
    """
    colors = []
    
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), 
                           cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    
    return colors


def draw_text_with_background(image: np.ndarray, text: str, 
                            position: Tuple[int, int], 
                            font_scale: float = 0.5,
                            font_color: Tuple[int, int, int] = (255, 255, 255),
                            bg_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    Draw text with background for better visibility
    
    Args:
        image: Image to draw on
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font size scale
        font_color: Text color (BGR)
        bg_color: Background color (BGR)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(image, 
                 (x - 2, y - text_height - 2),
                 (x + text_width + 2, y + baseline + 2),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)
