"""
ExpFlow - HPC Experiment Manager
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="expflow",
    version="0.4.0",
    author="Ali Hamza",
    author_email="ah7072@nyu.edu",
    description="Lightweight experiment tracking for HPC clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hurryingauto3/expflow-hpc",
    project_urls={
        "Bug Reports": "https://github.com/hurryingauto3/expflow-hpc/issues",
        "Documentation": "https://github.com/hurryingauto3/expflow-hpc/docs",
        "Source": "https://github.com/hurryingauto3/expflow-hpc",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "ai": ["google-generativeai>=0.3.0"],  # For Gemini API suggestions
        "dev": ["pytest>=7.0", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "expflow=expflow.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "expflow": ["examples/*.py"],
    },
    keywords="hpc slurm experiment-tracking machine-learning deep-learning research",
    zip_safe=False,
)
