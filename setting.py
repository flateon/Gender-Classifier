from pathlib import Path
from numpy import array

IMAGE_SIZE = (157, 224)
IMAGE_SUFFIX = 'png'
RAW_IMAGE_SUFFIX = 'jpg'
# 数据集位置
RAW_DATASET_PATH = Path('dataset/raw')
TRAINING_DATASET_PATH = Path('dataset/train')
VALIDATION_DATASET_PATH = Path('dataset/validation')
TESTING_DATASET_PATH = Path('dataset/test')
PREDICT_DATASET_PATH = Path('dataset/predict')
# 数据集比例
DATASET_RATIO = array((0.8, 0.1, 0.1))
