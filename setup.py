"""
Markov Logic Process (MLP) — setup configuration.
Install in editable mode for local development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="markov-logic-process",
    version="1.0.0",
    author="Saiyam Jain, Swaroop Bhowmik, Dipanjan Choudhury, Santosh Kumar Sahoo",
    author_email="saiyamjain@example.com",
    description=(
        "Markov Logic Process: Augmenting RL with Symbolic "
        "Association-Rule Reasoning via the Logos Module"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saiyamjain/markov-logic-process",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "gymnasium[box2d]>=0.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "mlxtend>=0.23.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "imageio>=2.31.0",
        "huggingface_hub>=0.19.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "ruff>=0.1.0",
            "imageio[ffmpeg]",
            "opencv-python-headless>=4.8.0",
        ],
        "video": [
            "imageio[ffmpeg]",
            "opencv-python-headless>=4.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=(
        "reinforcement-learning markov-decision-process association-rules "
        "symbolic-ai deep-q-network reward-shaping neuro-symbolic"
    ),
)
