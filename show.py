import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from model import get_model
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataset
from setting import *


def show_example(model_, dataloader_):
    figure = plt.figure(dpi=600)
    cols, rows = 15, 7
    cnt = 1
    with torch.no_grad():
        for images, labels in dataloader_:
            images, labels = images.to('cuda'), labels.to('cuda')
            predict_labels = model_(images)
            predicts = (predict_labels > 0).float()
            for image, label, predict in zip(images, labels, predicts):
                figure.add_subplot(rows, cols, cnt)
                plt.axis("off")
                plt.imshow(image.squeeze(0).permute(1, 2, 0).to('cpu'))
                plt.title('male' if predict == 1 else 'female', color='g' if predict == label else 'r', fontsize=5)
                cnt += 1
                if cnt > cols * rows:
                    break
            if cnt > cols * rows:
                break
    plt.savefig('examples.png', dpi=600)
    plt.show()


def show_error(model_, dataloader_):
    figure = plt.figure(dpi=600)
    cols, rows = 6, 4
    cnt = 1
    with torch.no_grad():
        for images, labels in dataloader_:
            images, labels = images.to('cuda'), labels.to('cuda')
            predict_labels = model_(images)
            predicts = (predict_labels > 0).float().squeeze()
            for idx in (predicts != labels).nonzero():
                figure.add_subplot(rows, cols, cnt)
                confidence = predict_labels[idx].tolist()[0]
                gender = 'female' if predicts[idx] == 0 else 'male'
                plt.title(gender + f'  f:{confidence[0]:.3f} m:{confidence[-1]:.3f}', color='r', fontsize=5)
                plt.axis("off")
                plt.imshow(images[idx].squeeze(0).permute(1, 2, 0).to('cpu'))
                cnt += 1
                if cnt > cols * rows:
                    break
            if cnt > cols * rows:
                break
    plt.savefig('error.png', dpi=600)
    plt.show()


def draw_scatter_plot(model_, dataloader_):
    figure = plt.figure(dpi=600)
    with torch.no_grad():
        for images, labels in dataloader_:
            predict_labels = model_(images.to('cuda')).cpu()
            colors = ['g' if x else 'r' for x in (predict_labels > 0).float().squeeze(1) == labels]
            plt.scatter(predict_labels[:, 0], predict_labels[:, 1], s=0.2, c=colors)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.savefig('scatter.png', dpi=600)
    plt.show()


def draw_hist(model_, dataloader_):
    figure = plt.figure(dpi=600)
    with torch.no_grad():
        for images, labels in dataloader_:
            predict_labels = model_(images.to('cuda')).cpu()
            plt.hist(predict_labels.squeeze().numpy(), bins=80, density=True, color='b')
    plt.savefig('hist.png', dpi=600)
    plt.show()


def show_pr_curve(model_, dataloader_, writer_):
    all_labels = []
    all_predict = []
    with torch.no_grad():
        for images, labels in dataloader_:
            all_predict.append((nn.Sigmoid()(model_(images.to('cuda')))).cpu())
            all_labels.append(labels)
    all_predict = torch.cat(all_predict).squeeze()
    all_labels = torch.cat(all_labels) == 1
    writer_.add_pr_curve('pr_curve', all_labels, all_predict, 0)
    writer_.close()


def show_model(model_, dataset_, writer_):
    images, _ = next(iter(dataset_))
    model_.to('cpu')
    writer_.add_graph(model_, images.unsqueeze(0))
    model_.to('cuda')
    writer_.close()


if __name__ == '__main__':
    writer = SummaryWriter('./runs/test')
    dataset = MyDataset(img_dir=TESTING_DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=250, num_workers=4, pin_memory=True)
    model = get_model().eval().to('cuda')
    model.load_state_dict(torch.load('model/resnet34_0.994131_08-29_20-25.pkl'))
    show_model(model, dataset, writer)
    show_example(model, dataloader)
    show_pr_curve(model, dataloader, writer)
    show_error(model, dataloader)
    draw_hist(model, dataloader)
