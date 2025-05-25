"""
Configuration file for text segmentation and transliteration system
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ModelConfig:
    """Configuration for the text detection model"""
    input_size: Tuple[int, int] = (300, 300)
    num_classes: int = 2  # background, text
    aspect_ratios: List[float] = None
    min_scale: float = 0.2
    max_scale: float = 0.9
    feature_maps: List[Tuple[int, int]] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    
    def __post_init__(self):
        if self.aspect_ratios is None:
            self.aspect_ratios = [1, 2, 3, 5, 7, 10]
        if self.feature_maps is None:
            self.feature_maps = [(38, 38), (19, 19), (10, 10), (5, 5)]


@dataclass
class OCRConfig:
    """Configuration for OCR settings"""
    languages: List[str] = None
    tesseract_cmd: str = None
    oem_mode: int = 3  # 0=Legacy, 1=LSTM, 2=Legacy+LSTM, 3=Default
    psm_mode: int = 8  # Page segmentation mode
    min_confidence: float = 30.0
    preprocessing: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['eng']
        
        # Set Tesseract command based on OS
        if self.tesseract_cmd is None:
            if os.name == 'nt':  # Windows
                self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            else:  # Linux/Mac
                self.tesseract_cmd = 'tesseract'


@dataclass
class TransliterationConfig:
    """Configuration for transliteration settings"""
    default_target_language: str = 'en'
    supported_languages: Dict[str, str] = None
    use_google_translate: bool = True
    use_transliterate_lib: bool = True
    cache_translations: bool = True
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = {
                'eng': 'en',
                'tam': 'ta',
                'hin': 'hi',
                'fra': 'fr',
                'deu': 'de',
                'spa': 'es',
                'por': 'pt',
                'rus': 'ru',
                'jpn': 'ja',
                'kor': 'ko',
                'ara': 'ar'
            }


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    max_image_size: int = 1920
    enhance_quality: bool = True
    auto_rotate: bool = False
    detect_orientation: bool = True
    batch_size: int = 8
    use_gpu: bool = True
    num_workers: int = 4


@dataclass
class SystemConfig:
    """Overall system configuration"""
    model: ModelConfig = None
    ocr: OCRConfig = None
    transliteration: TransliterationConfig = None
    processing: ProcessingConfig = None
    
    # Paths
    model_path: str = "models/text_detection_model.h5"
    output_dir: str = "results"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Logging
    log_level: str = "INFO"
    save_visualizations: bool = True
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.transliteration is None:
            self.transliteration = TransliterationConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        
        # Create directories
        for dir_path in [self.output_dir, self.log_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)


# Default configuration instance
default_config = SystemConfig()


# Hardware configuration for Raspberry Pi
raspberry_pi_config = SystemConfig(
    model=ModelConfig(
        confidence_threshold=0.6,  # Higher threshold for Pi
        nms_threshold=0.5
    ),
    processing=ProcessingConfig(
        max_image_size=1280,  # Smaller size for Pi
        use_gpu=False,
        num_workers=2,
        batch_size=4
    )
)


# Configuration presets
PRESETS = {
    'default': default_config,
    'raspberry_pi': raspberry_pi_config,
    'high_accuracy': SystemConfig(
        model=ModelConfig(
            confidence_threshold=0.7,
            nms_threshold=0.3
        ),
        ocr=OCRConfig(
            min_confidence=50.0,
            preprocessing=True
        )
    ),
    'fast_processing': SystemConfig(
        model=ModelConfig(
            confidence_threshold=0.4,
            feature_maps=[(19, 19), (10, 10)]  # Fewer feature maps
        ),
        processing=ProcessingConfig(
            max_image_size=800,
            enhance_quality=False,
            batch_size=16
        )
    )
}


def load_config(preset: str = 'default') -> SystemConfig:
    """
    Load configuration preset
    
    Args:
        preset: Name of configuration preset
        
    Returns:
        SystemConfig instance
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available presets: {list(PRESETS.keys())}")
    
    return PRESETS[preset]


def save_config(config: SystemConfig, filepath: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration instance
        filepath: Path to save configuration
    """
    import json
    
    config_dict = {
        'model': {
            'input_size': config.model.input_size,
            'num_classes': config.model.num_classes,
            'aspect_ratios': config.model.aspect_ratios,
            'min_scale': config.model.min_scale,
            'max_scale': config.model.max_scale,
            'confidence_threshold': config.model.confidence_threshold,
            'nms_threshold': config.model.nms_threshold
        },
        'ocr': {
            'languages': config.ocr.languages,
            'oem_mode': config.ocr.oem_mode,
            'psm_mode': config.ocr.psm_mode,
            'min_confidence': config.ocr.min_confidence,
            'preprocessing': config.ocr.preprocessing
        },
        'transliteration': {
            'default_target_language': config.transliteration.default_target_language,
            'use_google_translate': config.transliteration.use_google_translate,
            'use_transliterate_lib': config.transliteration.use_transliterate_lib,
            'cache_translations': config.transliteration.cache_translations
        },
        'processing': {
            'max_image_size': config.processing.max_image_size,
            'enhance_quality': config.processing.enhance_quality,
            'auto_rotate': config.processing.auto_rotate,
            'batch_size': config.processing.batch_size,
            'use_gpu': config.processing.use_gpu,
            'num_workers': config.processing.num_workers
        },
        'system': {
            'model_path': config.model_path,
            'output_dir': config.output_dir,
            'log_dir': config.log_dir,
            'cache_dir': config.cache_dir,
            'log_level': config.log_level,
            'save_visualizations': config.save_visualizations
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_file(filepath: str) -> SystemConfig:
    """
    Load configuration from file
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        SystemConfig instance
    """
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Create configuration instances
    model_config = ModelConfig(**config_dict.get('model', {}))
    ocr_config = OCRConfig(**config_dict.get('ocr', {}))
    trans_config = TransliterationConfig(**config_dict.get('transliteration', {}))
    proc_config = ProcessingConfig(**config_dict.get('processing', {}))
    
    # Create system config
    system_dict = config_dict.get('system', {})
    system_config = SystemConfig(
        model=model_config,
        ocr=ocr_config,
        transliteration=trans_config,
        processing=proc_config,
        **system_dict
    )
    
    return system_config
