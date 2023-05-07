import glob
import json
import os
import re
import subprocess
from bs4 import BeautifulSoup
from markdown import markdown
import pandas as pd


def download_repositories(repo_urls_file: str, repo_dir: str):
    """
    Download the Hugging Face repositories.
    """
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    with open(repo_urls_file, "r") as f:
        repositories_urls = json.load(f)["urls"]
        for url in repositories_urls:
            try:
                subprocess.run(["git", "clone", url], cwd=repo_dir)
            except subprocess.CalledProcessError as e:
                print("Command failed with error:", e.stderr)


def extract_markdown_from_directories(repo_urls_file: str, repo_dir: str, docs_dir: str):
    """
    This function reads markdown and markdownx files from the repositories directory,
    filters out non-English files, and adds the source GitHub URL as the first line of each file.
    The resulting files are saved in the docs_dir.
    """
    languages = pd.read_csv("language-codes.csv").loc[:,"alpha2"].tolist()
    languages.remove("en")

    files = glob.glob(repo_dir + "**/*.md", recursive=True)
    files += glob.glob(repo_dir + "**/*.mdx", recursive=True)

    repo_urls = []
    with open(repo_urls_file, "r") as f:
        repo_urls = json.load(f)["urls"]

    # filter out the files that are not in english
    filtered_files = []
    for filename in files:
        sep_file = filename.split("/")
        for seq in sep_file:
            if seq in languages:
                break
        else:
            filtered_files.append(filename)

    # generate a GitHub URL for a file based on its name and a list of possible repository URLs.
    def get_github_url(filename: str, repo_urls: str, repo_dir: str):
        source = filename.replace(repo_dir, '')
        repo_name, file_path = source.split('/', 1)
        repo_url_prefix = None
        for repo_url in repo_urls:
            if repo_name in repo_url:
                repo_url_prefix = repo_url
        if not repo_url_prefix:
            raise ValueError(f"Repo URL not found for {repo_name}")
        url = f'{repo_url_prefix}/blob/main/{file_path}' 
        return url

    # copy the files with the source added in the first line
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    for filename in filtered_files:
        source_url = get_github_url(filename, repo_urls, repo_dir)
        data = f"source: {source_url}\n\n"
        with open(filename, 'r') as f:
            data += f.read()
        with open(docs_dir + filename.split("/")[-1], 'w') as f:
            f.write(data)


def markdown_cleaner(data: str):
    """
    Clean markdown text.

    Args:
        data (str): The markdown text to be cleaned.

    Returns:
        str: The cleaned markdown text.
    """
    soupped = BeautifulSoup(markdown(data), "html.parser")
    raw_text = ''.join(soupped.findAll(string=True))
    clean_text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    # remove any special tokens e.g <|endoftext|>
    clean_text = re.sub(r"<\|endoftext\|>", "", clean_text, flags=re.DOTALL)
    # discard non english text
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", clean_text, flags=re.DOTALL)
    return "\n".join([t for t in clean_text.split("\n") if t])


if __name__ == '__main__':
    repo_urls_file = "./datasets/hf_repositories_urls.json"
    repo_dir = "./datasets/huggingface_repositories/"
    docs_dir = "./datasets/huggingface_docs/"
    download_repositories(repo_urls_file, repo_dir)
    extract_markdown_from_directories(repo_urls_file, repo_dir, docs_dir)
