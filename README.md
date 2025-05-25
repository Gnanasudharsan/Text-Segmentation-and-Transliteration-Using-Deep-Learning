# Text Segmentation and Transliteration Using Deep Learning

This project extracts and segments text from natural images, processes it using OCR, and enhances its usability through transliteration. It improves image-based text recognition for applications like multimedia indexing and assistance for the visually impaired.


## 🎯 Abstract

Reading text from images is crucial for understanding image content. This project aims to extract and segment text from natural images, process it through an OCR, and enhance its usability through transliteration. It aims to enhance image-based text recognition by segmenting text from natural images, processing it with OCR, and amplifying its utility through transliteration, catering to diverse applications like multimedia indexing and aiding the visually impaired.

## 🚀 Features

- **Text Detection**: Uses deep neural networks to detect text in natural images
- **Multi-orientation Support**: Detects text at various angles (except 90°)
- **OCR Integration**: Processes detected text using Tesseract OCR
- **Multi-language Support**: Recognizes English, Tamil, French, German, Dutch, and other languages
- **Transliteration**: Converts text from one script to another
- **Real-time Processing**: Efficient processing suitable for real-world applications

## 🛠️ System Architecture

The system consists of four main modules:

1. **Image Preprocessing**: Prepares input images for text detection
2. **Text Detection**: Uses a deep neural network based on VGG-16 architecture
3. **Text Recognition**: Employs Tesseract OCR for character recognition
4. **Transliteration**: Utilizes Google API for script conversion

### Neural Network Architecture

- 28-layer fully convolutional network
- 13 layers adopted from VGG-16
- 9 additional convolutional layers
- 6 text-box layers for multi-scale detection
- Non-maximum suppression (NMS) for post-processing

## 📋 Requirements

### Hardware Requirements
- Processor: Intel Core i5 or higher
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (optional but recommended)
- Storage: 10GB free space

### Software Requirements
- Python 3.8 or higher
- OpenCV 4.5+
- TensorFlow 2.x or PyTorch 1.x
- Tesseract OCR 4.0+
- Operating System: Ubuntu 18.04+ / Windows 10 / macOS 10.14+

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Gnanasudharsan/Text-Segmentation-and-Transliteration-Using-Deep-Learning.git
cd text-segmentation-transliteration
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`

## 💻 Usage

### Basic Text Detection and Recognition

```python
from text_detector import TextDetector
from text_recognizer import TextRecognizer

# Initialize detector and recognizer
detector = TextDetector()
recognizer = TextRecognizer()

# Process an image
image_path = "path/to/your/image.jpg"
detected_boxes = detector.detect(image_path)
recognized_text = recognizer.recognize(image_path, detected_boxes)

print(recognized_text)
```

### Text Transliteration

```python
from transliterator import Transliterator

# Initialize transliterator
trans = Transliterator()

# Transliterate text
original_text = "Hello World"
transliterated = trans.transliterate(original_text, dest_lang='hi')
print(transliterated)
```

### Complete Pipeline

```python
python main.py --input_image path/to/image.jpg --output_dir results/ --lang en --transliterate hi
```

## 📊 Model Performance

- **Text Detection Accuracy**: ~92% on standard datasets
- **OCR Accuracy**: 
  - Printed text: 95-98%
  - Scene text: 85-90%
- **Processing Speed**: ~0.5 seconds per image (on GPU)

## 🔍 Examples

### Successful Detection
- English text at various orientations
- Multi-language text (Tamil, French, German)
- Text with different fonts and sizes
- Slanted text up to 45 degrees

### Known Limitations
- Text at 90-degree angles
- Watermarked or transparent text
- Text with color similar to background
- Heavily occluded text
- Large character spacing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sri Sairam Engineering College for providing research facilities
- IEEE for publishing our work
- The open-source community for the tools and libraries used


---

**Note**: This is a research implementation. For production use, please ensure proper testing and optimization based on your specific requirements.
