import glob
import pandas as pd


def extract_markdown_from_directories():
    languages = pd.read_csv("./datasets/huggingface_docs/language-codes.csv").loc[:,"alpha2"].tolist()
    languages.remove("en")

    files = glob.glob('**/*.md', recursive=True) + glob.glob('**/*.mdx', recursive=True)
    filtered_files = []

    for file in files:
        sep_file = file.split('/')
        for seq in sep_file:
            if seq in languages:
                break
        else:
            filtered_files.append(file)

    # copy the files to /datasets/huggingface_docs/hf_filtered
    for file in filtered_files:
        with open(file, 'r') as f:
            data = f.read()
        print(f'./datasets/huggingface_docs/hf_filtered/{file.split("/")[-1:][0]}')
        with open(f'./datasets/huggingface_docs/hf_filtered/{file.split("/")[-1:][0]}', 'w') as f:
            f.write(data)


if __name__ == '__main__':
    extract_markdown_from_directories()
