# Hugging Face Question Answering Bot

This repository aims to develop a Hugging Face question answering bot that helps users develop own ML solutions, and troubleshoot technical issues related to the Hugging Face libraries. Our goal is to provide an efficient open-source solution.

# Table of Contents
- [Running bot](#running-bot)
- [Datasets](#dataset-list)
- [Development instructions](#development-intructions)

## Running Bot

1. Fill out the `.env` file with the required tokens.
2. Build and run the Docker container using the following commands:
```bash
docker build -t <container-name> .
docker run <container-name>
# or simply:
./run_docker.sh
```

## Dataset List

Below is a list of the datasets created during development:
- [Stack Overflow - Python](https://huggingface.co/datasets/KonradSzafer/stackoverflow_python_preprocessed)
- [Stack Overflow - Linux](https://huggingface.co/datasets/KonradSzafer/stackoverflow_linux)

## Development Instructions

To install all necessary Python packages, run the following command:

```bash
pip install -r requirements.txt
```
We use the pipreqsnb to generate the requirements.txt file. To install pipreqsnb, run the following command:

```bash
pip install pipreqsnb
```
To generate the requirements.txt file, run the following command:

```bash
pipreqsnb --force .
```
