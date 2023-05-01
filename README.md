# Hugging Face Question Answering Bot

This repository aims to develop a Hugging Face question answering bot that helps users develop own ML solutions, and troubleshoot technical issues related to the Hugging Face libraries. Our goal is to provide an efficient open-source solution.

# Table of Contents
- [Running bot](#running-bot)
    - [Running in a Docker](#running-in-a-docker)
    - [Running in a Python](#running-in-a-python)
- [Development instructions](#development-instructions)
- [Datasets](#dataset-list)

## Running Bot

### Running in a Docker
```bash
docker build -t <container-name> .
docker run <container-name>
# or simply:
./run_docker.sh
```

### Running in a Python
```bash
pip install -r requirements.txt
python3 -m bot
```

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

## Dataset List

Below is a list of the datasets created during development:
- [Stack Overflow - Python](https://huggingface.co/datasets/KonradSzafer/stackoverflow_python_preprocessed)
- [Stack Overflow - Linux](https://huggingface.co/datasets/KonradSzafer/stackoverflow_linux)
