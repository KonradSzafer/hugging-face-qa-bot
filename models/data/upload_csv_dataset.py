import sys
import pandas as pd
from datasets import Dataset, DatasetDict


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    filename = f'datasets/{dataset_name}.csv'
    df = pd.read_csv(filename)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'KonradSzafer/{dataset_name}', private=False)
