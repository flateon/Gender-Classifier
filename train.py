from datetime import datetime

import torch
import torch.nn as nn
from rich.progress import track
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import test
from model import get_model
from my_dataset import MyDataset
from setting import *

# 超参数
num_epochs = 15
batch_size = 90
learning_rate = 1e-4
step_size = 2


def main():
    accuracy = [0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 定义模型
    cnn = get_model()
    cnn.to(device).train()
    # cnn.load_state_dict(torch.load('model/resnet34_0.993000_2classes_08-20_21-14.pkl'))

    # 定义数据集与数据集加载器
    dataset, validation_dataset = MyDataset(img_dir=TRAINING_DATASET_PATH), MyDataset(img_dir=VALIDATION_DATASET_PATH)
    training_dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=250, num_workers=2, pin_memory=True)
    # 定义损失函数,优化算法及学习率调度器
    lose_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.333)
    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, labels) in track(enumerate(training_dataloader, start=1), total=len(dataset) / batch_size):
            predict_labels = cnn(images.to(device)).squeeze()
            loss = lose_fn(predict_labels, labels.to(device))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch:{epoch}, [{i * batch_size:^8}/{len(dataset):^8}] ,loss:{loss.item():.6f}')
        scheduler.step()
        # 验证模型在验证集上的准确度
        cnn.eval()
        current_acc = test.test_model(cnn, validation_dataloader)
        cnn.train()
        # 保存准确度提升的模型
        accuracy.append(current_acc)
        if current_acc >= max(accuracy):
            torch.save(cnn.state_dict(), f"model.pkl")
            print("save model")
            if current_acc > 0.98:
                torch.save(cnn.state_dict(),
                           f'model/resnet34_{current_acc:.6f}_2classes_{datetime.now().strftime("%m-%d_%H-%M")}.pkl')


if __name__ == '__main__':
    main()
