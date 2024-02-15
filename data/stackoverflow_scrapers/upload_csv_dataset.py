import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split



def main():
    dataset_name = sys.argv[1]
    test_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    print(f'dataset: {dataset_name}, test size: {test_size}')

    filename = f'datasets/{dataset_name}.csv'
    df = pd.read_csv(filename)
    dataset = Dataset.from_pandas(df)
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)
    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)
    dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
    dataset_dict.push_to_hub(f'KonradSzafer/{dataset_name}', private=False)


if __name__ == '__main__':
    main()
