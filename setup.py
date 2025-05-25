"""
Setup script for Text Segmentation and Transliteration package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-segmentation-transliteration",
    version="1.0.0",
    author="Meena P, Gnanasudharsan A, Sreevarshan S, Rithika S R, Guru Raghavendra S, S K UmaMaheswaran",
    author_email="meenuperiasamy3030@gmail.com",
    description="Text segmentation and transliteration using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-segmentation-transliteration",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "black>=21.6b0",
            "flake8>=3.9.2",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.1.1",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-process=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
)
