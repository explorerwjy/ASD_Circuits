#!/usr/bin/env python3
"""
Minimal setup.py for backward compatibility.
Modern configuration is in pyproject.toml.
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="patch_seq_transcriptome_mapping",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)
