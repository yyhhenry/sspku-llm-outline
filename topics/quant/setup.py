from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantization-lab",
    version="1.0.0",
    author="AI Research Lab",
    author_email="research@example.com",
    description="A comprehensive quantization lab for large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantization-lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "tensorboard>=2.13.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "plotly>=5.14.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "advanced": [
            "bitsandbytes>=0.41.0",
            "triton>=2.0.0",
            "sentencepiece>=0.1.99",
        ],
    },
)
