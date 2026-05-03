from setuptools import setup, find_packages

setup(
    name="adaptkv",
    version="0.1.0",
    author="Ali Hassan",
    author_email="22i-0541@nu.edu.pk",
    description="AdaptKV: Adaptive KV Cache Compression for Distributed LLM Inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "bitsandbytes>=0.43.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
