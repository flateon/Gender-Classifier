from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from test import test_model
from my_dataset import MyDataset
from setting import *
from model import get_model

# 超参数
NUM_EPOCHS = 12
BATCH_SIZE = 80
LEARNING_RATE = 3e-4
DECAY_STEP = 3
LR_DECAY_RATE = 0.3333


def train_(num_epochs, learning_rate, batch_size, lr_decay_rate, decay_steps):
    # 定义模型
    cnn = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn.to(device)
    # cnn.load_state_dict(torch.load('models/resnet34_0.993000_2classes_08-20_21-14.pkl'))
    # 定义数据集与数据集加载器
    dataset, validation_dataset = MyDataset(img_dir=TRAINING_DATASET_PATH), MyDataset(img_dir=VALIDATION_DATASET_PATH)
    training_dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=300, num_workers=2, pin_memory=True)
    # 定义损失函数,优化算法及学习率调度器
    lose_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=lr_decay_rate)
    writer = SummaryWriter(f'./runs/gender_classifier_{datetime.now().strftime("%m-%d_%H-%M")}')
    # 训练模型
    accuracy = [0, ]
    for epoch in range(num_epochs):
        cnn.train()
        program_bar = tqdm(total=len(training_dataloader), leave=False)
        for i, (images, labels) in enumerate(training_dataloader, start=1):
            predict_labels = cnn(images.to(device)).squeeze()
            loss = lose_fn(predict_labels, labels.to(device))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            program_bar.update()
            writer.add_scalar('training loss', loss.item(), epoch * len(training_dataloader) + i)
            if i % 10 == 0:
                tqdm.write(f'Epoch:{epoch}: [{i * batch_size:^8}/{len(dataset):^8}] ,loss:{loss.item():.6f}')
        scheduler.step()
        # 验证模型在验证集上的准确度
        current_acc = test_model(cnn, validation_dataloader)
        writer.add_scalar('validation acc', current_acc, epoch)
        # 保存准确度提升的模型
        if current_acc > max(accuracy):
            torch.save(cnn.state_dict(), f"model.pkl")
            print("save model")
            if current_acc > 0.99:
                torch.save(cnn.state_dict(),
                           f'models/resnet34_{current_acc:.6f}_{datetime.now().strftime("%m-%d_%H-%M")}.pkl')
        accuracy.append(current_acc)
    writer.close()
    del cnn
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train_(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, LR_DECAY_RATE, DECAY_STEP)
