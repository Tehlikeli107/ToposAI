from setuptools import setup, find_packages

setup(
    name="topos_ai",
    version="1.0.0",
    description="Kategori Teorisi ve Topos Mantığı Tabanlı Neuro-Symbolic Yapay Zeka Framework'ü",
    author="Topos AI Architect",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
        "yfinance>=0.2.0",
        "pandas>=1.3.0",
        "nltk>=3.6.0",
        "gradio>=3.0.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "requests>=2.25.0"
    ],
)
