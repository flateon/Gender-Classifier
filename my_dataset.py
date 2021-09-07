from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from setting import IMAGE_SUFFIX


class MyDataset(Dataset):
    def __init__(self, img_dir, transform=ToTensor()):
        self.img_dir = Path(img_dir)
        self.files = tuple(self.img_dir.glob('*.'+IMAGE_SUFFIX))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file)
        image = self.transform(image)
        label = 0.0 if file.stem.split('_')[0] == '女' else 1.0
        return image, label
