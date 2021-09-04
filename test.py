import torch
from model import get_model
from torch.utils.data import DataLoader

from my_dataset import MyDataset
from setting import *


def test_model(model, dataloader):
    with torch.no_grad():
        # 判断模型在GPU还是CPU
        model.eval()
        device = next(model.parameters()).device
        correct, total = 0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predict_label = (model(images) > 0).float().squeeze()
            correct += (predict_label == labels).sum().item()
            total += len(labels)
        print(f'The Accuracy of {total} images is {correct / total * 100:.6f}%')
    return correct / total


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model().to(device).eval()
    validation_dataloader = DataLoader(MyDataset(VALIDATION_DATASET_PATH), 400, num_workers=4, pin_memory=True)
    testing_dataloader = DataLoader(MyDataset(TESTING_DATASET_PATH), 400, num_workers=4, pin_memory=True)
    result = []
    # 加载模型参数
    for file in Path('models').glob('resnet34*.pkl'):
        model.load_state_dict(torch.load(file))
        # model.to('cpu')
        # torch.save(model, str(file).replace('.pkl', '.pth'))
        # model.to(device)
        print('Testing model:', file)
        result.append((file.name, test_model(model, testing_dataloader), test_model(model, validation_dataloader)))
    result.sort(key=lambda x: x[1] + x[2], reverse=True)
    for model in result:
        print(f'{model[0]}      Test:{model[1] * 100:.6f}%      Validation:{model[2] * 100:.6f}%')
