import glob
import json
import os
import re
import subprocess
from bs4 import BeautifulSoup
from markdown import markdown
import pandas as pd


def download_repositories(repositories_dir: str):
    if not os.path.exists(repositories_dir):
        os.makedirs(repositories_dir)
    with open("./datasets/hf_repositories_urls.json", "r") as f:
        repositories_urls = json.load(f)["urls"]
        for url in repositories_urls:
            try:
                subprocess.run(["git", "clone", url], cwd=repositories_dir)
            except subprocess.CalledProcessError as e:
                print("Command failed with error:", e.stderr)


def extract_markdown_from_directories(repositories_dir: str, documents_dir: str):
    languages = pd.read_csv("language-codes.csv").loc[:,"alpha2"].tolist()
    languages.remove("en")

    files = glob.glob(repositories_dir + "**/*.md", recursive=True)
    files += glob.glob(repositories_dir + "**/*.mdx", recursive=True)

    # filter out the files that are not in english
    filtered_files = []
    for filename in files:
        sep_file = filename.split("/")
        for seq in sep_file:
            if seq in languages:
                break
        else:
            filtered_files.append(filename)

    # copy the files with the source added in the first line
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    for filename in filtered_files:
        data = f"source: {filename.replace(repositories_dir, '')}\n\n"
        with open(filename, 'r') as f:
            data += f.read()
        with open(documents_dir + filename.split("/")[-1], 'w') as f:
            f.write(data)


def markdown_cleaner(data: str):
    soupped = BeautifulSoup(markdown(data), "html.parser")
    raw_text = ''.join(soupped.findAll(string=True))
    clean_text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    # remove any special tokens e.g <|endoftext|>
    clean_text = re.sub(r"<\|endoftext\|>", "", clean_text, flags=re.DOTALL)
    # discard non english text
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", clean_text, flags=re.DOTALL)
    return "\n".join([t for t in clean_text.split("\n") if t])


if __name__ == '__main__':
    repositories_dir = "./datasets/huggingface_repositories/"
    documents_dir = "./datasets/huggingface_docs/"
    download_repositories(repositories_dir)
    extract_markdown_from_directories(repositories_dir, documents_dir)
