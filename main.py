"""
Main script for Text Segmentation and Transliteration
Combines detection, recognition, and transliteration in a complete pipeline
"""

import argparse
import os
import cv2
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict

from text_detector import TextDetector
from text_recognizer import TextRecognizer
from transliterator import Transliterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextProcessingPipeline:
    """Complete text processing pipeline"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the pipeline
        
        Args:
            model_path: Path to pre-trained detector model
        """
        logger.info("Initializing text processing pipeline...")
        self.detector = TextDetector(model_path)
        self.recognizer = TextRecognizer()
        self.transliterator = Transliterator()
        logger.info("Pipeline initialized successfully")
    
    def process_image(self, image_path: str, 
                     target_language: str = None,
                     visualize: bool = True) -> Dict:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path: Path to input image
            target_language: Target language for transliteration
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Detect text regions
        logger.info("Detecting text regions...")
        detections = self.detector.detect(image_path)
        logger.info(f"Found {len(detections)} text regions")
        
        # Recognize text
        logger.info("Recognizing text...")
        recognized_texts = self.recognizer.recognize(image_path, detections)
        logger.info(f"Recognized {len(recognized_texts)} text segments")
        
        # Transliterate if target language specified
        transliterations = []
        if target_language:
            logger.info(f"Transliterating to {target_language}...")
            for text_obj in recognized_texts:
                result = self.transliterator.transliterate(
                    text_obj.text, 
                    target_language
                )
                transliterations.append(result)
        
        # Create result dictionary
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'detections': [
                {
                    'box': det['box'],
                    'score': det['score']
                }
                for det in detections
            ],
            'recognized_texts': [
                {
                    'text': text.text,
                    'confidence': text.confidence,
                    'box': text.box,
                    'language': text.language
                }
                for text in recognized_texts
            ],
            'transliterations': [
                {
                    'original': trans.original_text,
                    'transliterated': trans.transliterated_text,
                    'source_lang': trans.source_language,
                    'target_lang': trans.target_language,
                    'method': trans.method
                }
                for trans in transliterations
            ] if transliterations else []
        }
        
        # Visualize if requested
        if visualize:
            vis_image = self.visualize_results(image_path, results)
            results['visualization'] = vis_image
        
        return results
    
    def visualize_results(self, image_path: str, results: Dict) -> str:
        """
        Create visualization of processing results
        
        Args:
            image_path: Original image path
            results: Processing results
            
        Returns:
            Path to visualization image
        """
        image = cv2.imread(image_path)
        
        # Draw detection boxes
        for i, detection in enumerate(results['detections']):
            box = detection['box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text if available
            if i < len(results['recognized_texts']):
                text = results['recognized_texts'][i]['text']
                # Draw background for text
                cv2.rectangle(image, (x1, y1-20), (x2, y1), (0, 255, 0), -1)
                cv2.putText(image, text[:20] + '...' if len(text) > 20 else text, 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save visualization
        output_path = image_path.replace('.', '_result.')
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def process_batch(self, image_paths: List[str], 
                     target_language: str = None,
                     output_dir: str = None) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            target_language: Target language for transliteration
            output_dir: Directory to save results
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}")
            try:
                result = self.process_image(image_path, target_language)
                results.append(result)
                
                # Save individual result if output directory specified
                if output_dir:
                    result_path = os.path.join(
                        output_dir, 
                        f"result_{Path(image_path).stem}.json"
                    )
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Text Segmentation and Transliteration Pipeline'
    )
    
    parser.add_argument(
        '--input_image', '-i',
        type=str,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--input_dir', '-d',
        type=str,
        help='Directory containing images to process'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    parser.add_argument(
        '--lang', '-l',
        type=str,
        default='eng',
        help='Language for OCR (default: eng)'
    )
    
    parser.add_argument(
        '--transliterate', '-t',
        type=str,
        help='Target language for transliteration'
    )
    
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        help='Path to pre-trained model weights'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Create visualization of results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = TextProcessingPipeline(args.model_path)
    
    # Set OCR language
    pipeline.recognizer.set_languages([args.lang])
    
    # Process images
    if args.input_image:
        # Single image
        result = pipeline.process_image(
            args.input_image, 
            args.transliterate,
            args.visualize
        )
        
        # Save result
        output_path = os.path.join(args.output_dir, 'result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\n=== Processing Summary ===")
        print(f"Image: {args.input_image}")
        print(f"Detected regions: {len(result['detections'])}")
        print(f"Recognized texts: {len(result['recognized_texts'])}")
        
        if result['recognized_texts']:
            print("\n--- Recognized Text ---")
            for text in result['recognized_texts']:
                print(f"Text: {text['text']}")
                print(f"Confidence: {text['confidence']:.1f}%")
                print()
        
        if result['transliterations']:
            print("\n--- Transliterations ---")
            for trans in result['transliterations']:
                print(f"Original: {trans['original']}")
                print(f"Transliterated: {trans['transliterated']}")
                print()
    
    elif args.input_dir:
        # Batch processing
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(Path(args.input_dir).glob(ext))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        results = pipeline.process_batch(
            [str(p) for p in image_paths],
            args.transliterate,
            args.output_dir
        )
        
        # Save summary
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_images': len(results),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': len([r for r in results if 'error' in r]),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
