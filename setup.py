from setuptools import setup, find_packages

setup(
    name="topos_ai",
    version="0.1.0",
    description="Kategori Teorisi ve Topos Mantığı Tabanlı Neuro-Symbolic Yapay Zeka Framework'ü",
    author="Topos AI Architect",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0"
    ],
)
