import os

from PIL import Image
from numpy import floor
from setting import *
import random


def main():
    files = list(DATASET_PATH.glob("*.jpg"))
    training_dataset_size, validation_dataset_size, testing_dataset_size = floor(len(files) * DATASET_RATIO)
    random.shuffle(files)
    for idx, file in enumerate(files, start=1):
        image = Image.open(file).resize((IMAGE_HEIGHT, IMAGE_WIDTH))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if idx < training_dataset_size:
            image.save(TRAINING_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')
        elif idx < training_dataset_size + validation_dataset_size:
            image.save(VALIDATION_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')
        else:
            image.save(TESTING_DATASET_PATH / f'{file.stem}.{IMAGE_SUFFIX}')
        if idx % 10 == 0:
            print(idx)


if __name__ == '__main__':
    main()
