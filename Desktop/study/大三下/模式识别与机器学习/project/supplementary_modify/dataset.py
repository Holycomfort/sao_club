from PIL import Image
from torch.utils.data import Dataset
from jaccard import get_all_area
import numpy as np
import sys


class TxtLoader(Dataset):
    def __init__(self, txt_path, transform_all, transform_data):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs 
        self.transform_all = transform_all
        self.transform_data = transform_data

    def __getitem__(self, index):
        fn, lb = self.imgs[index]
        # transform all
        img = np.asarray(Image.open(fn)).reshape((628, 628, 1))
        label = np.asarray(Image.open(lb)).reshape((628, 628, 1))
        # center net
        #label = centerize(label.reshape((628, 628))).reshape((628, 628, 1))
        comb = np.concatenate([img, label, label], axis=2)
        comb = Image.fromarray(comb.astype('uint8')).convert('RGB')
        comb = self.transform_all(comb)
        comb = np.asarray(comb)
        img = Image.fromarray(comb[:,:, 0]) # Image, L
        label = np.array(comb[:,:, 1])  # np.array, Len*Len
        label[label > 0] = 1       

        # transform data
        img = self.transform_data(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


def get_center_point(mask):
    areas = get_all_area(mask)
    center = {}
    for i,j in areas.items():
        if len(j) < 50:
            continue
        center[i] = np.array(j).mean(axis=0)
        center[i] = (int(center[i][0]),int(center[i][1]))
    return center

def dist_square(t1,t2):
    return max((t1[0]-t2[0],t2[0]-t1[0],t1[1]-t2[1],t2[1]-t1[1]))

def find_min_dist(center):
    min_dist = dict(zip(center.keys(),[10000]*len(center)))
    for i,m in center.items():
        for j,k in center.items():
            if j==i:
                continue
            min_dist[i] = min(min_dist[i],dist_square(m,k))
    return min_dist

def centerize(mask):
    center = get_center_point(mask)
    min_dist = find_min_dist(center)
    l = np.zeros(mask.shape,dtype=np.uint8)
    for i,j in center.items():
        x,y = j
        r = int(min_dist[i]**0.5)
        l[x-r:x+r+1,y-r:y+r+1] = i
    return l
