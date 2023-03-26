import os
import glob
import json
import subprocess
import pandas as pd
import re
from bs4 import BeautifulSoup
from markdown import markdown


def download_repositories():
    repositories_dir = "datasets/huggingface_repositories"
    if not os.path.exists(repositories_dir):
        os.makedirs(repositories_dir)
    with open("datasets/hf_repositories_urls.json", "r") as f:
        repositories_urls = json.load(f)["urls"]
        for url in repositories_urls:
            try:
                subprocess.run(["git", "clone", url], cwd=repositories_dir)
            except subprocess.CalledProcessError as e:
                print("Command failed with error:", e.stderr)


def extract_markdown_from_directories():
    languages = pd.read_csv("language-codes.csv").loc[:,"alpha2"].tolist()
    languages.remove("en")

    files = glob.glob('./datasets/huggingface_repositories/**/*.md', recursive=True) + glob.glob('**/*.mdx', recursive=True)
    filtered_files = []

    for file in files:
        sep_file = file.split('/')
        for seq in sep_file:
            if seq in languages:
                break
        else:
            filtered_files.append(file)
    data = ""
    # copy the files to /datasets/huggingface_docs/hf_filtered
    for file in filtered_files:
        with open(file, 'r') as f:
            data += markdown_cleaner(f.read())
    for i in range(0, len(data), 512*32):
        with open(f"datasets/huggingface_docs/hf_filtered_{i/(512*32)}.txt", "w") as f:
            f.write(data[i:i+512*32])


def markdown_cleaner(data: str):
    soupped = BeautifulSoup(markdown(data), "html.parser")
    raw_text = ''.join(soupped.findAll(text=True))
    clean_text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    return "\n".join([t for t in clean_text.split("\n") if t])

if __name__ == '__main__':
    download_repositories()
    extract_markdown_from_directories()
