import random

from PIL import Image
from numpy import floor
from tqdm import tqdm

from setting import *


def main():
    TRAINING_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    VALIDATION_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    TESTING_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    files = list(RAW_DATASET_PATH.glob(f"*.{RAW_IMAGE_SUFFIX}"))
    training_dataset_size, validation_dataset_size, testing_dataset_size = floor(len(files) * DATASET_RATIO)
    random.shuffle(files)
    for idx, file in tqdm(enumerate(files, start=1), total=len(files)):
        image = Image.open(file).resize(IMAGE_SIZE)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if idx < training_dataset_size:
            image.save(TRAINING_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')
        elif idx < training_dataset_size + validation_dataset_size:
            image.save(VALIDATION_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')
        else:
            image.save(TESTING_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')


if __name__ == '__main__':
    main()
