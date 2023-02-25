from datetime import datetime
from datasets import load_dataset
from bs4 import BeautifulSoup


def preprocess_dataset():
    dataset = load_dataset('koutch/stackoverflow_python', split='train')
    dataset = dataset.filter(
        lambda example:
            example['question_score'] > 100 and
            example['answer_score'] > 5 and
            datetime.strptime(example['answer_date'], '%Y-%m-%dT%H:%M:%SZ').year > 2010
        )

    def html2text(example):
        soup = BeautifulSoup(example, 'html.parser')
        return ''.join(soup.findAll(string=True))

    def transform_questions(example):
        example['question_body'] = html2text(example['question_body'])
        return example

    def transform_answers(example):
        example['answer_body'] = html2text(example['answer_body'])
        return example

    dataset = dataset.map(lambda example: transform_questions(example))
    dataset = dataset.map(lambda example: transform_answers(example))
    dataset = dataset.remove_columns([
        'question_score', 'question_date', 'question_id',
        'answer_date', 'answer_id', 'answer_score', 'tags',
    ])
    return dataset


def show_info(dataset):
    print(dataset.info, '\n')
    print(f'dataset len: {len(dataset)}')
    print(f"example question: {dataset[0]['question_body']}")
    print(f"example answer: {dataset[0]['answer_body']}")


if __name__ == '__main__':
    dataset = preprocess_dataset()
    dataset.push_to_hub('KonradSzafer/stackoverflow_python_preprocessed', private=False)
    show_info(dataset)
