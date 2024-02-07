import os
import shutil
import subprocess
import warnings
from pathlib import Path


COMMON_FILES = ['.git', 'README.md', __file__.split('/')[-1]]


def remove_old_files():
    filenames = os.listdir('./')
    filenames = [f for f in filenames if f not in COMMON_FILES]
    for file_path in filenames:
        p = Path(file_path)
        if p.exists():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)


def clone_repository():
    repo_url = 'https://github.com/KonradSzafer/hugging-face-qa-bot.git'
    subprocess.run(['git', 'clone', '--depth', '1', repo_url])


def copy_files():
    src = './hugging-face-qa-bot'
    for item in COMMON_FILES:
        full_path = os.path.join(src, item)
        if os.path.isfile(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
    for item in Path(src).iterdir():
        shutil.move(str(item), '.')
    shutil.rmtree(src)


def main():
    path = os.getcwd().lower()
    current_dir = path.split('/')[-1]
    if current_dir != 'hugging-face-qa-bot':
        print('Updating the HF space...')
        remove_old_files()
        clone_repository()
        copy_files()
    else:
        warnings.warn('You are in the hugging-face-qa-bot reposotory')


if __name__ == '__main__':
    main()
   