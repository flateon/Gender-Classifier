from pathlib import Path
from numpy import array

IMAGE_HEIGHT = 153
IMAGE_WIDTH = 218
IMAGE_SUFFIX = 'png'
# 数据集位置
DATASET_PATH = Path('dataset/all')
TRAINING_DATASET_PATH = Path('dataset/train')
VALIDATION_DATASET_PATH = Path('dataset/validation')
TESTING_DATASET_PATH = Path('dataset/test')
PREDICT_DATASET_PATH = Path('dataset/predict')
# 数据集比例
DATASET_RATIO = array((0.8, 0.1, 0.1))
