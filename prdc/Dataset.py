import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        self.root = data_root
        self.transforms = transforms
        self.imgs = os.listdir(self.root)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.root, self.imgs[index])).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
        except:
            print(f"Error in image {self.imgs[index]}")
            # rurn a black image
            img = torch.zeros(3, 224, 224)
        return img