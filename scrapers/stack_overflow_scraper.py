import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


def scrape_question_with_answers(url: str):
    print(url)


def scrape_questions_page(url: str, min_votes: int=1, min_answers: int=1):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    posts_summaries = soup.find_all('div', {'class':'s-post-summary js-post-summary'})

    for summary in posts_summaries:
        stats_div = summary.find('div', {'class': 's-post-summary--stats'})
        vote_div = stats_div.find('div', {
            'class': 's-post-summary--stats-item s-post-summary--stats-item__emphasized',
            'title': re.compile(r'^Score of \d+$')})
        if vote_div:
            vote_number = int(vote_div.find('span', {'class': 's-post-summary--stats-item-number'}).text)
        else:
            vote_number = 0
        answer_div = stats_div.find('div', {
            'class': 's-post-summary--stats-item',
            'title': re.compile(r'^\d+ answers$')})
        if answer_div:
            answer_number = int(answer_div.find('span', {'class': 's-post-summary--stats-item-number'}).text)
        else:
            answer_number = 0

        question_href = summary.find('a', {'class': 's-link'})['href']
        if vote_number >= min_votes and answer_number >= min_answers:
            scrape_question_with_answers(question_href)


def crawl_question_pages(base_url: str, n_pages: int=10):
    for num in range(n_pages):
        url = base_url.format(num)
        scrape_questions_page(url)


if __name__ == '__main__':
    url = 'https://stackoverflow.com/questions/tagged/python?tab=votes&page={}&pagesize=15'
    crawl_question_pages(url)
