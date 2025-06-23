"""
Setup configuration for Les Audits-Affaires Evaluation Harness
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="les-audits-affaires-eval",
    version="1.0.0",
    author="LegML Team",
    author_email="contact@legml.ai",
    description="Framework d'évaluation pour les LLM sur le benchmark juridique français Les Audits-Affaires",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legmlai/les-audits-affaires-eval-harness",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: French",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "les-audits-eval=scripts.run_evaluation:main",
            "les-audits-test=scripts.test_setup:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="llm evaluation french legal benchmark jurisprudence",
    project_urls={
        "Bug Reports": "https://github.com/legmlai/les-audits-affaires-eval-harness/issues",
        "Source": "https://github.com/legmlai/les-audits-affaires-eval-harness",
        "Documentation": "https://github.com/legmlai/les-audits-affaires-eval-harness/blob/main/README.md",
        "Dataset": "https://huggingface.co/datasets/legmlai/les-audits-affaires",
    },
) 