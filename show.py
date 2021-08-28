import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from model import get_model
from my_dataset import MyDataset
from setting import *


def show_example(_model, _dataloader):
    figure = plt.figure(dpi=600)
    cols, rows = 15, 7
    cnt = 1
    with torch.no_grad():
        for images, labels in _dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            predict_labels = model(images)
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


def show_error(_model, _dataloader):
    figure = plt.figure(dpi=600)
    cols, rows = 6, 4
    cnt = 1
    with torch.no_grad():
        for images, labels in _dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            predict_labels = model(images)
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


def draw_scatter_plot(_model, _dataloader):
    figure = plt.figure(dpi=600)
    with torch.no_grad():
        for images, labels in _dataloader:
            predict_labels = model(images.to('cuda')).cpu()
            colors = ['g' if x else 'r' for x in (predict_labels > 0).float().squeeze(1) == labels]
            plt.scatter(predict_labels[:, 0], predict_labels[:, 1], s=0.2, c=colors)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.savefig('scatter.png', dpi=600)
    plt.show()


def draw_hist(_model, _dataloader):
    figure = plt.figure(dpi=600)
    with torch.no_grad():
        for images, labels in _dataloader:
            predict_labels = model(images.to('cuda')).cpu()
            plt.hist(predict_labels.squeeze().numpy(), bins=100, density=True, color='b')
    plt.savefig('hist.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    dataset = MyDataset(img_dir=TESTING_DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=250, shuffle=True, num_workers=4, pin_memory=True)
    model = get_model().eval().to('cuda')
    model.load_state_dict(torch.load('model/resnet34_0.990000_08-15_18-41.pkl'))
    show_example(model, dataloader)
    show_error(model, dataloader)
    draw_hist(model, dataloader)
