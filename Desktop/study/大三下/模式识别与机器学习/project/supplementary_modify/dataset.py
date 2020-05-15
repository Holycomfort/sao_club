from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class TxtLoader(Dataset):
    def __init__(self, txt_path, transform):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs 
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('L') 
        img = self.transform(img)
        label = np.asarray(Image.open(label).resize((256, 256)))
        label = np.array(label).astype(np.int16)
        label[label > 0] = 1
        return img, label

    def __len__(self):
        return len(self.imgs)
