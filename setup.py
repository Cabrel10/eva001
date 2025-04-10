from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="Morningstar",
    version="0.1.0",
    author="Morningstar Team",
    author_email="contact@morningstar-trading.com", 
    description="Advanced crypto trading framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/morningstar-trading/Morningstar",
    packages=find_packages(include=["Morningstar*"]),
    package_dir={"": "."},
    package_data={
        "Morningstar": ["configs/*.py", "data/*.parquet"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "morningstar=Morningstar.cli:main",
        ],
    },
)
