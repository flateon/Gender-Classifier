import sys
from pathlib import Path
from time import time

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, img_dir, recursion=True, img_suffix='jpg', size=(153, 218), transform=ToTensor()):
        self.img_dir = Path(img_dir)
        self.size = size
        if recursion:
            self.files = tuple(self.img_dir.rglob(f'*.{img_suffix}'))
        else:
            self.files = tuple(self.img_dir.glob(f'*.{img_suffix}'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file)
        if image.size[0] > image.size[1]:  # 宽图
            file_name = str(file) + '#'
        else:
            file_name = str(file)
        image = self.transform(image.resize(self.size).convert('RGB'))
        return image, file_name


def predictor(model, dataloader_, path_, image_num, move_file=True):
    with torch.no_grad():
        start = time()
        cnt = 0
        bar = tqdm(total=image_num)
        for images, files in dataloader_:
            predict_labels = model(images)
            for file, predict in zip(files, predict_labels):
                if file[-1] != '#':
                    file = Path(file)
                    gender = '男' if predict > 0 else '女'
                    tqdm.write(f'{file.name:16} {gender} {predict.item():.4f}')
                    dst_path = path_ / gender / file.name
                else:
                    file = Path(file[:-1])
                    dst_path = path_ / '其他' / file.name
                if move_file:
                    try:
                        file.rename(dst_path)
                    except:
                        tqdm.write(f'文件同名:{file}')
                cnt += 1
            bar.update(len(images))
        tqdm.write(f'成功分类图片{cnt}张,耗时{time() - start:.2f}秒')


def main(model_path='model.pth', batch_size=20, num_workers=0, size=(153, 218)):
    while True:
        cnn = torch.load(model_path)
        print('基于深度神经网络的照片性别分类器'.center(60, '-'))
        while True:
            path = Path(input('请输入待分类文件或文件夹路径:'))
            if path.exists() and path.is_dir():
                while True:
                    recursion = input('是否递归子文件夹(yes/no):')
                    if recursion == 'yes':
                        recursion = True
                        break
                    elif recursion == 'no':
                        recursion = False
                        break
                    else:
                        print('请输入"yes"或"no"')
                (path / '男').mkdir(parents=True, exist_ok=True)
                (path / '女').mkdir(parents=True, exist_ok=True)
                (path / '其他').mkdir(parents=True, exist_ok=True)
                dataset = MyDataset(path, recursion=recursion, size=size)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
                predictor(cnn, dataloader, path, len(dataset), move_file=True)
                break
            elif path.exists() and path.is_file():
                image = ToTensor()(Image.open(path).resize(size).convert('RGB')).unsqueeze(0)
                predictor(cnn, [(image, [str(path)])], path, 1, move_file=False)
                break
            else:
                print('路径不存在,请重试')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        main()
