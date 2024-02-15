from datetime import datetime
from datasets import load_dataset
from bs4 import BeautifulSoup


def preprocess_dataset():
    """
    Preprocesses the 'koutch/stackoverflow_python' dataset.

    Returns:
        datasets.arrow_dataset.Dataset: The preprocessed dataset.
    """
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

    def transforms(example):
        example['answer'] = html2text(example['answer_body'])
        example['question'] = html2text(example['question_body'])
        return example

    dataset = dataset.map(lambda example: transforms(example))
    dataset = dataset.remove_columns([
        'question_score', 'question_date', 'question_id',
        'answer_date', 'answer_id', 'answer_score', 'tags',
        'question_body', 'answer_body'
    ])
    return dataset


def show_info(dataset):
    """
    Print information about the dataset.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset.
    """
    print(dataset.info, '\n')
    print(f'dataset len: {len(dataset)}')
    print(f"example question: {dataset[0]['question']}")
    print(f"example answer: {dataset[0]['answer']}")


if __name__ == '__main__':
    dataset = preprocess_dataset()
    dataset.push_to_hub('KonradSzafer/stackoverflow_python_preprocessed', private=False)
    show_info(dataset)
