import os
from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mobilenerf-edge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Scene recreation on edge devices using MobileNeRF and OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mobilenerf-edge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mobilenerf-train=mobilenerf_edge.scripts.train:main",
            "mobilenerf-optimize=mobilenerf_edge.scripts.optimize_model:main",
            "mobilenerf-deploy=mobilenerf_edge.scripts.run_edge_demo:main",
        ],
    },
)
