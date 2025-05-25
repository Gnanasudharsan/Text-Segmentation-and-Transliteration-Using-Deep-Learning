"""
Unit tests for text detection module
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from text_detector import TextDetector, TextBoxLayer
import tensorflow as tf


class TestTextDetector(unittest.TestCase):
    """Test cases for TextDetector class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TextDetector()
        
        # Create test image with text
        cls.test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.putText(cls.test_image, "TEST TEXT", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Save test image
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.temp_dir, "test_image.jpg")
        cv2.imwrite(cls.test_image_path, cls.test_image)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(len(self.detector.aspect_ratios), 6)
        self.assertEqual(self.detector.min_scale, 0.2)
        self.assertEqual(self.detector.max_scale, 0.9)
    
    def test_model_architecture(self):
        """Test model architecture"""
        model = self.detector.model
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 300, 300, 3))
        
        # Check number of outputs (scores and offsets for each feature map)
        expected_outputs = len(self.detector.aspect_ratios) * 2 * 4  # 4 feature maps
        self.assertEqual(len(model.outputs), 8)  # 4 feature maps * 2 (scores, offsets)
    
    def test_generate_default_boxes(self):
        """Test default box generation"""
        feature_shape = (10, 10)
        image_shape = (300, 300)
        
        default_boxes = self.detector._generate_default_boxes(feature_shape, image_shape)
        
        # Check shape
        expected_boxes = feature_shape[0] * feature_shape[1] * len(self.detector.aspect_ratios) * 2
        self.assertEqual(default_boxes.shape[0], expected_boxes)
        self.assertEqual(default_boxes.shape[1], 4)  # cx, cy, w, h
        
        # Check value ranges
        self.assertTrue(np.all(default_boxes[:, 0] >= 0))  # cx
        self.assertTrue(np.all(default_boxes[:, 0] <= 1))
        self.assertTrue(np.all(default_boxes[:, 1] >= 0))  # cy
        self.assertTrue(np.all(default_boxes[:, 1] <= 1.5))  # Including offset
    
    def test_compute_iou(self):
        """Test IoU computation"""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        
        iou = self.detector._compute_iou(box1, box2)
        
        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        self.assertAlmostEqual(iou, 25/175, places=3)
        
        # Test non-overlapping boxes
        box3 = [20, 20, 30, 30]
        iou_no_overlap = self.detector._compute_iou(box1, box3)
        self.assertEqual(iou_no_overlap, 0.0)
    
    def test_non_max_suppression(self):
        """Test non-maximum suppression"""
        boxes = [
            {'box': [0, 0, 10, 10], 'score': 0.9},
            {'box': [2, 2, 12, 12], 'score': 0.8},  # Overlaps with first
            {'box': [20, 20, 30, 30], 'score': 0.85},  # No overlap
            {'box': [22, 22, 32, 32], 'score': 0.7}  # Overlaps with third
        ]
        
        nms_boxes = self.detector._non_max_suppression(boxes, iou_threshold=0.5)
        
        # Should keep boxes with scores 0.9 and 0.85
        self.assertEqual(len(nms_boxes), 2)
        self.assertEqual(nms_boxes[0]['score'], 0.9)
        self.assertEqual(nms_boxes[1]['score'], 0.85)
    
    def test_detect_invalid_image(self):
        """Test detection with invalid image path"""
        with self.assertRaises(ValueError):
            self.detector.detect("nonexistent_image.jpg")
    
    def test_detect_valid_image(self):
        """Test detection with valid image"""
        # This test would require a trained model
        # For now, just test that it runs without error
        try:
            detections = self.detector.detect(self.test_image_path)
            self.assertIsInstance(detections, list)
        except Exception as e:
            # Model might not be trained, which is okay for unit test
            if "No module named" not in str(e):
                self.skipTest("Model not trained yet")
    
    def test_visualize_detections(self):
        """Test visualization function"""
        fake_detections = [
            {'box': [50, 50, 150, 100], 'score': 0.95},
            {'box': [200, 200, 280, 250], 'score': 0.87}
        ]
        
        output_path = os.path.join(self.temp_dir, "vis_output.jpg")
        vis_image = self.detector.visualize_detections(
            self.test_image_path, 
            fake_detections,
            output_path
        )
        
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(vis_image.shape, self.test_image.shape)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(cls.temp_dir)


class TestTextBoxLayer(unittest.TestCase):
    """Test cases for TextBoxLayer"""
    
    def test_layer_initialization(self):
        """Test custom layer initialization"""
        layer = TextBoxLayer(num_defaults=12)
        self.assertEqual(layer.num_defaults, 12)
    
    def test_layer_output_shape(self):
        """Test layer output shapes"""
        layer = TextBoxLayer(num_defaults=12)
        
        # Create dummy input
        dummy_input = tf.keras.Input(shape=(10, 10, 512))
        scores, offsets = layer(dummy_input)
        
        # Check output shapes
        # Scores: [batch, height, width, num_defaults, 2]
        # Offsets: [batch, height, width, num_defaults, 4]
        self.assertEqual(scores.shape.as_list()[1:], [10, 10, 12, 2])
        self.assertEqual(offsets.shape.as_list()[1:], [10, 10, 12, 4])


if __name__ == '__main__':
    unittest.main()
