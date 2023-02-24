import re
import csv
import requests
from typing import List
import pandas as pd
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
            qa_data.append(scrape_question_with_answers(question_href))
    return qa_data


def crawl_and_save_qa(
    filename: str,
    base_url: str,
    n_pages: int=10,
    min_votes: int=1,
    min_answers: int=1
):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for page_num in range(n_pages):
            page_data = scrape_questions_page(
                base_url.format(page_num),
                min_votes,
                min_answers
            )
            for qa_data in page_data:
                writer.writerow(qa_data)


if __name__ == '__main__':
    filename = 'stack_overflow_python.csv'
    url = 'https://stackoverflow.com/questions/tagged/python?tab=votes&page={}&pagesize=15'
    crawl_and_save_qa(
        filename=filename,
        base_url=url,
        n_pages=1,
        min_votes=1,
        min_answers=1
    )
