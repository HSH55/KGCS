"""KGCS package configuration."""
from setuptools import setup, find_packages

setup(
    name="kgcs",
    version="1.0.0",
    description="KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection in Aerial Images",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Wei Hu, Suhang Hu, Fei Ma, Qihao Zhao, Fan Zhang",
    url="https://github.com/HSH55/KGCS",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "tqdm",
        "requests",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="object-detection, zero-shot, aerial-imagery, sam, clip, remote-sensing",
)
