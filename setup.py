from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="topos_ai",
    version="1.0.0",
    description="A neuro-symbolic AI framework bridging Category Theory with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tehlikeli107/ToposAI",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "experiments*", "applications*", "benchmarks*"]),
    install_requires=[
        "torch>=2.0.0",
        "networkx>=2.5",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",
        "requests>=2.25.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "full": [
            "yfinance>=0.2.0",
            "pandas>=1.3.0",
            "nltk>=3.6.0",
            "gradio>=3.0.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "tensorboard>=2.10.0",
            "scikit-learn>=1.0.0",
            "triton>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    entry_points={
        "console_scripts": [
            "topos-train=train_custom_llm:train",
            "topos-chat=chat_custom_llm:chat_with_topos",
            "topos-showcase=topos_showcase:run_showcase",
        ]
    },
)
