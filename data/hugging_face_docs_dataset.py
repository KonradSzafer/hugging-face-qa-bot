import glob
import json
import os
import re
import subprocess
from typing import List

import requests
import pandas as pd
from bs4 import BeautifulSoup
from markdown import markdown
import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor, ClearOutputPreprocessor
from tqdm import tqdm


VALIDATE_URLS = False


def download_repositories(repo_urls_file: str, repo_dir: str):
    """
    Downloads the Hugging Face repositories.
    """
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    with open(repo_urls_file, "r") as f:
        repositories_urls = json.load(f)["urls"]
        print(f'Downloading {len(repositories_urls)} repositories')
        for url in repositories_urls:
            try:
                subprocess.run(["git", "clone", url], cwd=repo_dir)
            except subprocess.CalledProcessError as e:
                print("Command failed with error:", e.stderr)


class EmptyCellPreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        if cell.source.strip() == '':
            cell.source = ''
            cell.cell_type = 'raw'
        return cell, resources


def convert_notebook_to_txt(filename: str):
    """
    Converts a notebook to a markdown file.
    """
    with open(filename) as f:
        notebook = nbformat.read(f, as_version=4)
    # id validation error fix
    for cell in notebook['cells']:
        cell['id'] = str(cell['id'])

    clear_output = ClearOutputPreprocessor()
    notebook, resources = clear_output.preprocess(notebook, {})

    exporter = MarkdownExporter()
    exporter.register_preprocessor(EmptyCellPreprocessor, enabled=True)
    output_notebook_text, resources = exporter.from_notebook_node(notebook)

    new_filename = filename.replace('.ipynb', '_ipynb.txt')
    with open(new_filename, 'w') as f:
        f.write(output_notebook_text)
    return new_filename


def extract_files_from_directories(
    repo_urls_file: str,
    repo_dir: str,
    docs_dir: str,
    files_extensions: List[str]
) -> None:

    """
    This function reads markdown and markdownx files from the repositories directory,
    filters out non-English files, and adds the source GitHub URL as the first line of each file.
    The resulting files are saved in the docs_dir.
    """
    languages = pd.read_csv("language-codes.csv").loc[:,"alpha2"].tolist()
    languages.remove("en")

    files = [
        filename
        for extension in files_extensions
        for filename in glob.glob(repo_dir + f"**/*{extension}", recursive=True)
    ]
    print(f'Used extensions: {", ".join(files_extensions)}')
    print(f'Found {len(files)} files')

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
    print(f'Found {len(filtered_files)} files in English')

    # generate a GitHub URL for a file based on its name and a list of possible repository URLs
    def get_github_url(filename: str, repo_urls: str, repo_dir: str) -> str:
        source = filename.replace(repo_dir, '')
        repo_name, file_path = source.split('/', 1)
        repo_url_prefix = None
        for repo_url in repo_urls:
            if repo_name == repo_url.split('/')[-1]:
                repo_url_prefix = repo_url
                break
        if not repo_url_prefix:
            raise ValueError(f"Repo URL not found for {repo_name}")
        url = f'{repo_url_prefix}/blob/main/{file_path}'
        if VALIDATE_URLS:
            try:
                response = requests.get(url)
                response.raise_for_status()
            except:
                print(f'filename: {filename}')
                print(f'repo: {repo_name}, file: {file_path}')
                print(f'url: {url}')
                raise
        return url

    # creates a valid filename by replacing certain characters and removing the repo_dir path
    def create_filename_from_path(filename: str, repo_dir: str) -> str:
        filename = filename.replace(repo_dir, '')
        chars_to_replace = ['/', '{', '}', '-', '.']
        filename = ''.join(['_' if c in chars_to_replace else c for c in filename])
        return filename

    # copy the files with the source added in the first line
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    copied_files = []
    for filename in tqdm(filtered_files):
        source_url = get_github_url(filename, repo_urls, repo_dir)
        data = f"source: {source_url}\n\n"
        # convert jupyter notebooks to txt files
        try:
            if filename.endswith('.ipynb'):
                filename = convert_notebook_to_txt(filename)
            # rename and copy files
            with open(filename, 'r') as f:
                data += f.read()
            output_filename = docs_dir + create_filename_from_path(filename, repo_dir)
            with open(output_filename, 'w') as f:
                f.write(data)
            if not os.path.isfile(output_filename):
                raise ValueError(f"Failed to create the output file: {output_filename}")
            copied_files.append(output_filename)
        except Exception as ex:
            print(f'Failed to copy file {filename}: {ex}')

    print(f'Successfully copied {len(set(copied_files))}/{len(filtered_files)} files')


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
    extract_files_from_directories(
        repo_urls_file, repo_dir, docs_dir,
        files_extensions=['.md', '.mdx', '.ipynb']
    )
