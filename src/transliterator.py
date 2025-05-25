"""
Text Transliteration Module
Converts text from one script to another using various APIs and libraries
"""

import logging
from typing import Dict, List, Optional
from googletrans import Translator
from transliterate import translit, get_available_language_codes
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransliterationResult:
    """Data class for transliteration results"""
    original_text: str
    transliterated_text: str
    source_language: str
    target_language: str
    method: str  # 'googletrans' or 'transliterate'


class Transliterator:
    """Main transliteration class supporting multiple methods"""
    
    def __init__(self):
        """Initialize transliterator with available methods"""
        self.google_translator = Translator()
        self.available_transliterate_langs = get_available_language_codes()
        
        # Language code mappings
        self.lang_mappings = {
            'eng': 'en',
            'tam': 'ta',
            'hin': 'hi',
            'fra': 'fr',
            'deu': 'de',
            'rus': 'ru',
            'ell': 'el',  # Greek
            'ara': 'ar',
            'jpn': 'ja',
            'kor': 'ko',
            'zho': 'zh-cn'
        }
        
        logger.info(f"Initialized transliterator with {len(self.available_transliterate_langs)} "
                   f"transliterate languages")
    
    def transliterate(self, text: str, 
                     target_language: str,
                     source_language: str = None) -> TransliterationResult:
        """
        Transliterate text to target script
        
        Args:
            text: Input text to transliterate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TransliterationResult object
        """
        if not text:
            return TransliterationResult(
                original_text=text,
                transliterated_text='',
                source_language=source_language or 'unknown',
                target_language=target_language,
                method='none'
            )
        
        # Try transliterate library first (for supported languages)
        if target_language in self.available_transliterate_langs:
            try:
                transliterated = translit(text, target_language)
                return TransliterationResult(
                    original_text=text,
                    transliterated_text=transliterated,
                    source_language=source_language or 'auto',
                    target_language=target_language,
                    method='transliterate'
                )
            except Exception as e:
                logger.warning(f"Transliterate failed: {e}")
        
        # Fall back to Google Translate
        try:
            # Convert language codes if needed
            target_code = self.lang_mappings.get(target_language, target_language)
            
            if source_language:
                source_code = self.lang_mappings.get(source_language, source_language)
                result = self.google_translator.translate(text, 
                                                         dest=target_code,
                                                         src=source_code)
            else:
                result = self.google_translator.translate(text, dest=target_code)
            
            return TransliterationResult(
                original_text=text,
                transliterated_text=result.text,
                source_language=result.src,
                target_language=target_language,
                method='googletrans'
            )
            
        except Exception as e:
            logger.error(f"Google Translate failed: {e}")
            return TransliterationResult(
                original_text=text,
                transliterated_text=text,  # Return original if all fails
                source_language=source_language or 'unknown',
                target_language=target_language,
                method='failed'
            )
    
    def batch_transliterate(self, texts: List[str],
                          target_language: str,
                          source_language: str = None) -> List[TransliterationResult]:
        """
        Transliterate multiple texts
        
        Args:
            texts: List of texts to transliterate
            target_language: Target language for all texts
            source_language: Source language (auto-detect if None)
            
        Returns:
            List of transliteration results
        """
        results = []
        
        for text in texts:
            result = self.transliterate(text, target_language, source_language)
            results.append(result)
        
        return results
    
    def transliterate_to_multiple(self, text: str,
                                target_languages: List[str],
                                source_language: str = None) -> Dict[str, TransliterationResult]:
        """
        Transliterate text to multiple target languages
        
        Args:
            text: Input text
            target_languages: List of target language codes
            source_language: Source language (auto-detect if None)
            
        Returns:
            Dictionary mapping language codes to results
        """
        results = {}
        
        for target_lang in target_languages:
            result = self.transliterate(text, target_lang, source_language)
            results[target_lang] = result
        
        return results
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        try:
            detection = self.google_translator.detect(text)
            return detection.lang
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'unknown'
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get list of supported languages for each method
        
        Returns:
            Dictionary with supported languages per method
        """
        return {
            'transliterate': self.available_transliterate_langs,
            'googletrans': list(self.lang_mappings.values())
        }
    
    def transliterate_with_fallback(self, text: str,
                                  target_languages: List[str],
                                  source_language: str = None) -> Optional[TransliterationResult]:
        """
        Try transliteration with multiple target languages until one succeeds
        
        Args:
            text: Input text
            target_languages: List of target languages to try
            source_language: Source language
            
        Returns:
            First successful transliteration result
        """
        for target_lang in target_languages:
            result = self.transliterate(text, target_lang, source_language)
            if result.method != 'failed':
                return result
        
        return None


# Specialized transliterators for specific language pairs
class TamilTransliterator:
    """Specialized transliterator for Tamil script"""
    
    def __init__(self):
        self.tamil_to_latin_map = {
            'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii',
            'உ': 'u', 'ஊ': 'uu', 'எ': 'e', 'ஏ': 'ee',
            'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
            'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஞ': 'nya',
            'ட': 'ta', 'ண': 'na', 'த': 'tha', 'ந': 'nha',
            'ப': 'pa', 'ம': 'ma', 'ய': 'ya', 'ர': 'ra',
            'ல': 'la', 'வ': 'va', 'ழ': 'zha', 'ள': 'la',
            'ற': 'ra', 'ன': 'na'
        }
    
    def transliterate_to_latin(self, tamil_text: str) -> str:
        """Convert Tamil script to Latin script"""
        result = []
        for char in tamil_text:
            if char in self.tamil_to_latin_map:
                result.append(self.tamil_to_latin_map[char])
            elif char.isspace():
                result.append(char)
            else:
                result.append(char)  # Keep unknown characters
        
        return ''.join(result)


class CustomTransliterator:
    """Base class for custom transliteration rules"""
    
    def __init__(self):
        self.rules = {}
    
    def add_rule(self, source: str, target: str):
        """Add a transliteration rule"""
        self.rules[source] = target
    
    def apply_rules(self, text: str) -> str:
        """Apply transliteration rules to text"""
        result = text
        for source, target in self.rules.items():
            result = result.replace(source, target)
        return result
