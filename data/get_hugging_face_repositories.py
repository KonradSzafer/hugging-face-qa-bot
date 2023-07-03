import json
import argparse
import requests
from typing import List


def get_repositories_names(token):
    url = f'https://api.github.com/orgs/huggingface/repos?per_page=1000'
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        repos = json.loads(response.content)
        repo_names = [
            repo['full_name'] for repo in repos
            if repo['stargazers_count'] >= 100
        ]
        return repo_names
    else:
        return 'Error: '+str(response.status_code)


def save_repositories_urls(repositories_names: List[str], output_filename: str):
    urls = ['https://github.com/'+repo_name for repo_name in repositories_names]
    data = {"urls": urls}
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str)
    args = parser.parse_args()
    repositories = get_repositories_names(token=args.token)
    save_repositories_urls(repositories, 'datasets/hf_repositories_urls_scraped.json')
