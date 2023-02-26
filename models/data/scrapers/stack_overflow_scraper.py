import re
import csv
import time
import requests
from typing import List
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def scrape_question_with_answers(question_url: str) -> List[str]:
    url = 'https://stackoverflow.com/' + question_url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('title').text.replace(' - Stack Overflow', '')
    question_div = soup.find('div', {'class': 'postcell post-layout--right'})
    question = question_div.find('p').text
    answers_div = soup.find('div', {'class': 'answercell post-layout--right'})
    answer = answers_div.find('div', {'class': 's-prose js-post-body'}).text
    return [title, question, answer, url]


def scrape_questions_page(url: str, min_votes: int, min_answers: int) -> List[List[str]]:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    posts_summaries = soup.find_all('div', {'class':'s-post-summary js-post-summary'})

    qa_data = []
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
            try:
                qa_data.append(scrape_question_with_answers(question_href))
            except Exception as error:
                print(error)

        time.sleep(1.5)
    return qa_data


def crawl_and_save_qa(
    filename: str,
    base_url: str,
    start_page: int,
    n_pages: int=10,
    min_votes: int=1,
    min_answers: int=1
):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if start_page == 1:
            writer.writerow(['title', 'question', 'answer', 'url'])
        for page_num in tqdm(range(start_page, start_page+n_pages)):
            page_data = scrape_questions_page(
                base_url.format(page_num),
                min_votes,
                min_answers
            )
            if page_data:
                for qa_data in page_data:
                    writer.writerow(qa_data)


if __name__ == '__main__':
    filename = '../datasets/stackoverflow_linux.csv'
    url = 'https://stackoverflow.com/questions/tagged/linux?tab=votes&page={}&pagesize=15'
    crawl_and_save_qa(
        filename=filename,
        base_url=url,
        start_page=21,
        n_pages=10,
        min_votes=1,
        min_answers=1
    )
