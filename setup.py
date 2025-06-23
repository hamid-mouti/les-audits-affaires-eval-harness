"""
Setup configuration for Les Audits-Affaires Evaluation Harness
"""

from setuptools import setup, find_packages
import os
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover – fallback for older Pythons
    import tomli as tomllib

# Extract metadata from pyproject.toml so that version bumps in one place propagate everywhere
_pyproject_path = Path(__file__).with_name("pyproject.toml")
if _pyproject_path.exists():
    _pyproject_data = tomllib.loads(_pyproject_path.read_text())
    _project_meta = _pyproject_data.get("project", {})
    _PACKAGE_NAME = _project_meta.get("name", "les-audits-affaires-eval-harness")
    _VERSION = _project_meta.get("version", "0.0.0")
else:
    # Fallback values; this should rarely happen during normal development
    _PACKAGE_NAME = "les-audits-affaires-eval-harness"
    _VERSION = "0.0.0"

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name=_PACKAGE_NAME,
    version=_VERSION,
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