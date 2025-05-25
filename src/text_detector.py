"""
Text Detection Module using Deep Neural Networks
Based on VGG-16 architecture with text-box layers
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextBoxLayer(layers.Layer):
    """Custom text-box layer for multi-scale text detection"""
    
    def __init__(self, num_defaults=12, **kwargs):
        super(TextBoxLayer, self).__init__(**kwargs)
        self.num_defaults = num_defaults
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.num_defaults * 6,  # 2 for scores + 4 for offsets
            kernel_size=(1, 5),
            padding='same'
        )
        super(TextBoxLayer, self).build(input_shape)
        
    def call(self, inputs):
        outputs = self.conv(inputs)
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Reshape to separate scores and offsets
        outputs = tf.reshape(outputs, [batch_size, height, width, self.num_defaults, 6])
        
        scores = outputs[..., :2]
        offsets = outputs[..., 2:]
        
        return scores, offsets


class TextDetector:
    """Main text detection class using deep neural networks"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the text detector
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model = self._build_model()
        if model_path:
            self.load_weights(model_path)
        
        # Default box configurations
        self.aspect_ratios = [1, 2, 3, 5, 7, 10]
        self.min_scale = 0.2
        self.max_scale = 0.9
        
    def _build_model(self) -> Model:
        """Build the text detection model based on VGG-16"""
        # Load VGG-16 base model
        base_model = VGG16(include_top=False, input_shape=(300, 300, 3))
        
        # Get intermediate layers from VGG-16
        conv1_1 = base_model.get_layer('block1_conv1').output
        conv2_2 = base_model.get_layer('block2_conv2').output
        conv3_3 = base_model.get_layer('block3_conv3').output
        conv4_3 = base_model.get_layer('block4_conv3').output
        
        # Add extra convolutional layers
        conv5_3 = layers.Conv2D(512, 3, padding='same', activation='relu')(conv4_3)
        pool5 = layers.MaxPooling2D(2, strides=2, padding='same')(conv5_3)
        
        conv6 = layers.Conv2D(1024, 3, padding='same', activation='relu')(pool5)
        conv7 = layers.Conv2D(1024, 1, padding='same', activation='relu')(conv6)
        
        conv8_1 = layers.Conv2D(256, 1, padding='same', activation='relu')(conv7)
        conv8_2 = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(conv8_1)
        
        conv9_1 = layers.Conv2D(128, 1, padding='same', activation='relu')(conv8_2)
        conv9_2 = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(conv9_1)
        
        # Add text-box layers
        text_box_layers = []
        feature_maps = [conv4_3, conv7, conv8_2, conv9_2]
        
        for i, feature_map in enumerate(feature_maps):
            text_box = TextBoxLayer(num_defaults=len(self.aspect_ratios) * 2, 
                                   name=f'text_box_{i}')
            scores, offsets = text_box(feature_map)
            text_box_layers.append((scores, offsets))
        
        # Create model
        model = Model(inputs=base_model.input, 
                     outputs=[item for sublist in text_box_layers for item in sublist])
        
        return model
    
    def _generate_default_boxes(self, feature_shape: Tuple[int, int], 
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate default boxes for a feature map"""
        height, width = feature_shape
        img_height, img_width = image_shape
        
        boxes = []
        for y in range(height):
            for x in range(width):
                cx = (x + 0.5) / width
                cy = (y + 0.5) / height
                
                for ratio in self.aspect_ratios:
                    # Horizontal boxes
                    w = self.min_scale * np.sqrt(ratio)
                    h = self.min_scale / np.sqrt(ratio)
                    boxes.append([cx, cy, w, h])
                    
                    # Vertically offset boxes
                    cy_offset = cy + 0.5 / height
                    if cy_offset <= 1.0:
                        boxes.append([cx, cy_offset, w, h])
        
        return np.array(boxes)
    
    def _decode_predictions(self, predictions: List[np.ndarray], 
                           default_boxes: List[np.ndarray]) -> List[Dict]:
        """Decode model predictions to bounding boxes"""
        decoded_boxes = []
        
        for i in range(0, len(predictions), 2):
            scores = predictions[i]
            offsets = predictions[i + 1]
            defaults = default_boxes[i // 2]
            
            # Apply sigmoid to scores
            scores = 1 / (1 + np.exp(-scores))
            
            # Decode offsets
            for j in range(scores.shape[0]):
                for k in range(scores.shape[1]):
                    for l in range(scores.shape[2]):
                        score = scores[j, k, l, 1]  # Text class score
                        
                        if score > 0.5:  # Confidence threshold
                            default = defaults[k * scores.shape[1] + l]
                            offset = offsets[j, k, l]
                            
                            # Decode box
                            cx = default[0] + default[2] * offset[0]
                            cy = default[1] + default[3] * offset[1]
                            w = default[2] * np.exp(offset[2])
                            h = default[3] * np.exp(offset[3])
                            
                            # Convert to corner format
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            decoded_boxes.append({
                                'box': [x1, y1, x2, y2],
                                'score': float(score)
                            })
        
        return decoded_boxes
    
    def _non_max_suppression(self, boxes: List[Dict], 
                            iou_threshold: float = 0.45) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping boxes"""
        if not boxes:
            return []
        
        # Sort by score
        boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            boxes = [box for box in boxes 
                    if self._compute_iou(current['box'], box['box']) < iou_threshold]
        
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute intersection over union of two boxes"""
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
    
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect text in an image
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected text boxes with scores
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_shape = image.shape[:2]
        resized = cv2.resize(image, (300, 300))
        normalized = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        # Run detection
        predictions = self.model.predict(batch)
        
        # Generate default boxes
        default_boxes = []
        feature_shapes = [(38, 38), (19, 19), (10, 10), (5, 5)]
        for shape in feature_shapes:
            defaults = self._generate_default_boxes(shape, (300, 300))
            default_boxes.append(defaults)
        
        # Decode predictions
        detected_boxes = self._decode_predictions(predictions, default_boxes)
        
        # Apply NMS
        detected_boxes = self._non_max_suppression(detected_boxes)
        
        # Scale boxes back to original image size
        scale_x = original_shape[1] / 300
        scale_y = original_shape[0] / 300
        
        for box in detected_boxes:
            box['box'][0] *= scale_x
            box['box'][1] *= scale_y
            box['box'][2] *= scale_x
            box['box'][3] *= scale_y
        
        logger.info(f"Detected {len(detected_boxes)} text regions")
        return detected_boxes
    
    def visualize_detections(self, image_path: str, 
                           detections: List[Dict], 
                           output_path: str = None) -> np.ndarray:
        """
        Visualize detected text boxes on the image
        
        Args:
            image_path: Path to the original image
            detections: List of detected boxes
            output_path: Path to save the visualization
            
        Returns:
            Image with drawn boxes
        """
        image = cv2.imread(image_path)
        
        for detection in detections:
            box = detection['box']
            score = detection['score']
            
            # Draw box
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw score
            label = f"{score:.2f}"
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def load_weights(self, path: str):
        """Load pre-trained model weights"""
        self.model.load_weights(path)
        logger.info(f"Loaded weights from {path}")
    
    def save_weights(self, path: str):
        """Save model weights"""
        self.model.save_weights(path)
        logger.info(f"Saved weights to {path}")
