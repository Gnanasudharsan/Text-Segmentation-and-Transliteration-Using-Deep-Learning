"""
Text Recognition Module using Tesseract OCR
Processes detected text regions and extracts text content
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecognizedText:
    """Data class for recognized text results"""
    text: str
    confidence: float
    box: List[float]
    language: str


class TextRecognizer:
    """Text recognition using Tesseract OCR"""
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the text recognizer
        
        Args:
            languages: List of languages to recognize (e.g., ['eng', 'tam'])
        """
        self.languages = languages or ['eng']
        self.tesseract_config = '--oem 3 --psm 8'
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found. Please install Tesseract OCR: {e}")
    
    def preprocess_region(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for better OCR results
        
        Args:
            image: Input image region
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(binary, 3)
        
        # Resize if too small
        height, width = denoised.shape
        if height < 30:
            scale = 30 / height
            new_width = int(width * scale)
            denoised = cv2.resize(denoised, (new_width, 30), 
                                interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def extract_text_region(self, image: np.ndarray, 
                          box: List[float]) -> np.ndarray:
        """
        Extract text region from image using bounding box
        
        Args:
            image: Full image
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image region
        """
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Add padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def recognize_text(self, image_region: np.ndarray, 
                      language: str = None) -> Dict:
        """
        Recognize text in an image region
        
        Args:
            image_region: Cropped image region containing text
            language: Language code for recognition
            
        Returns:
            Dictionary with recognized text and confidence
        """
        lang = language or '+'.join(self.languages)
        
        # Preprocess the region
        processed = self.preprocess_region(image_region)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed)
        
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                pil_image, 
                lang=lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Valid detection
                    text_parts.append(data['text'][i])
                    confidences.append(int(data['conf'][i]))
            
            text = ' '.join(text_parts).strip()
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'language': lang
            }
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'language': lang
            }
    
    def recognize(self, image_path: str, 
                 detections: List[Dict]) -> List[RecognizedText]:
        """
        Recognize text from all detected regions
        
        Args:
            image_path: Path to the original image
            detections: List of detected text boxes
            
        Returns:
            List of recognized text objects
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        results = []
        
        for detection in detections:
            box = detection['box']
            
            # Extract region
            region = self.extract_text_region(image, box)
            
            # Recognize text
            ocr_result = self.recognize_text(region)
            
            if ocr_result['text']:  # Only add non-empty results
                result = RecognizedText(
                    text=ocr_result['text'],
                    confidence=ocr_result['confidence'],
                    box=box,
                    language=ocr_result['language']
                )
                results.append(result)
                
                logger.info(f"Recognized: '{result.text}' "
                          f"(confidence: {result.confidence:.1f}%)")
        
        return results
    
    def recognize_with_multiple_languages(self, image_region: np.ndarray,
                                        languages: List[str]) -> Dict:
        """
        Try recognition with multiple languages and return best result
        
        Args:
            image_region: Image region to recognize
            languages: List of language codes to try
            
        Returns:
            Best recognition result
        """
        best_result = None
        best_confidence = 0
        
        for lang in languages:
            result = self.recognize_text(image_region, lang)
            
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
        
        return best_result
    
    def post_process_text(self, text: str, language: str = 'eng') -> str:
        """
        Post-process recognized text to fix common errors
        
        Args:
            text: Raw recognized text
            language: Language code
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Language-specific processing
        if language == 'eng':
            # Fix common OCR errors
            replacements = {
                '0': 'O',  # Zero to O in certain contexts
                '1': 'I',  # One to I in certain contexts
                '5': 'S',  # Five to S in certain contexts
            }
            
            # Apply replacements contextually
            words = text.split()
            for i, word in enumerate(words):
                if word.isalpha() and any(char in word for char in '015'):
                    for old, new in replacements.items():
                        word = word.replace(old, new)
                    words[i] = word
            
            text = ' '.join(words)
        
        return text
    
    def batch_recognize(self, image_path: str,
                       detections_list: List[List[Dict]]) -> List[List[RecognizedText]]:
        """
        Batch process multiple images
        
        Args:
            image_path: List of image paths
            detections_list: List of detection results for each image
            
        Returns:
            List of recognition results for each image
        """
        all_results = []
        
        for img_path, detections in zip(image_path, detections_list):
            results = self.recognize(img_path, detections)
            all_results.append(results)
        
        return all_results
    
    def set_languages(self, languages: List[str]):
        """Update the languages for recognition"""
        self.languages = languages
        logger.info(f"Updated languages to: {languages}")
    
    def get_available_languages(self) -> List[str]:
        """Get list of available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.error(f"Could not get available languages: {e}")
            return []
