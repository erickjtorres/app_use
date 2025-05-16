#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

# Get the current directory
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get version
def find_version():
    with open(os.path.join("app_use", "__init__.py"), encoding="utf-8") as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Core dependencies
install_requires = [
    "dart-vm-client==0.0.1.dev4",
    "pydantic>=2.0.0",
    "python-dotenv>=0.21.0",
    "langchain-core>=0.1.0",
]

# Optional dependencies
extra_requires = {
    "openai": ["langchain-openai>=0.0.1", "openai>=1.0.0"],
    "groq": ["langchain-groq>=0.0.1", "groq>=0.4.0"],
    "google": ["google-generativeai>=0.3.0", "langchain-google-vertexai>=0.0.1"],
    "anthropic": ["anthropic>=0.5.0", "langchain-anthropic>=0.0.1"],
    "all": ["langchain-openai>=0.0.1", "openai>=1.0.0", 
            "langchain-groq>=0.0.1", "groq>=0.4.0", 
            "google-generativeai>=0.3.0", "langchain-google-vertexai>=0.0.1",
            "anthropic>=0.5.0", "langchain-anthropic>=0.0.1"],
}

setup(
    name="app_use",
    version=find_version(),
    description="A library for controlling Flutter applications via the Dart VM Service",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="App Use Developers",
    author_email="info@app-use.dev",
    url="https://github.com/app-use/app-use",
    packages=find_packages(include=["app_use", "app_use.*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extra_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ],
    keywords="flutter, dart, testing, automation, ai, vm, dart-vm-service",
) 