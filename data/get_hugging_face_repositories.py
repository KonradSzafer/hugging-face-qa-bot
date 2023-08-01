import json
import argparse
import requests
from typing import List


def get_repositories_names(token: str, min_stars: int) -> List[str]:
    repos_per_page = 100
    repo_names = []
    i = 0
    while True:
        url = \
            f'https://api.github.com/orgs/huggingface/repos?' \
            f'per_page={repos_per_page}&page={i}'
        headers = {'Authorization': f'token {token}'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            repos = json.loads(response.content)
            repo_names += [
                repo['full_name'] for repo in repos
                if repo['stargazers_count'] >= min_stars
            ]
            if len(repos) < repos_per_page:
                break
            i += 1
        else:
            return 'Error: '+str(response.status_code)
    return list(set(repo_names))


def save_repositories_urls(repositories_names: List[str], output_filename: str):
    urls = ['https://github.com/'+repo_name for repo_name in repositories_names]
    data = {"urls": urls}
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str)
    parser.add_argument('--stars', type=str)
    args = parser.parse_args()
    repositories = get_repositories_names(token=args.token, min_stars=int(args.stars))
    repositories += [
        'huggingface/hf-endpoints-documentation',
        'gradio-app/gradio'
    ]
    save_repositories_urls(repositories, 'datasets/hf_repositories_urls_scraped.json')
