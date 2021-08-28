from pathlib import Path

PATH = Path(r'')


def add_gender_index(path, gender):
    for idx, file in enumerate(path.glob('*.jpg')):
        # if len(file.name.split('_')) <= 2:
        name = f'{gender}_{file.stem}_{idx}{file.suffix}'
        file.rename(path / name)


def add_index(path):
    for idx, file in enumerate(path.glob('*.jpg')):
        if len(file.stem.split('_')) <= 2:
            name = f'{file.stem}_{idx}{file.suffix}'
            file.rename(path / name)


def remove_index(path):
    for file in path.glob('*.jpg'):
        name = f'{file.stem.split("_")[0]}{file.suffix}'
        file.rename(path / name)


def remove_gender_index(path):
    for file in path.glob('*.jpg'):
        name = f'{file.stem.split("_")[1]}{file.suffix}'
        file.rename(path / name)


if __name__ == '__main__':
    add_gender_index(PATH, '男')
