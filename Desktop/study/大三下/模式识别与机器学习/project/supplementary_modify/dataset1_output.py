from __future__ import absolute_import

import cv2
from PIL import Image
import imageio
import numpy as np
import os, sys
import os.path as osp
import models, models2
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torchvision import transforms
from jaccard import get_all_area
from dataset import unfold


def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))

def img_standardization(img):
    img = unit16b2uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')
    
def load_images(file_names):
    images = []
    for file_name in file_names:
        img = cv2.imread(file_name, -1)
        img = img_standardization(img)
        images.append(img)
    return images


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 


class BinaryThresholding:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        gray = bgr_to_gray(img)
        (_, binary_mask) = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.medianBlur(binary_mask, 5)
        connectivity = 4
        _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity, cv2.CV_32S)
        return label_img

class UnetSeg:
    def __init__(self, l, lc):
        unet = nn.DataParallel(models2.U_Net(1, 1))
        unet.load_state_dict(torch.load("./best_model_x.pt"))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        unet = unet.to(device)
        unet.eval()
        cnet = nn.DataParallel(models2.U_Net(1, 1))
        cnet.load_state_dict(torch.load("./best_model_x_c.pt"))
        cnet = cnet.to(device)
        cnet.eval()
        self.l, self.lc = l, lc
        self.unet = unet
        self.cnet = cnet
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(l),
            transforms.ToTensor()
        ])
        self.transform_c = transforms.Compose([
            transforms.Resize(lc),
            transforms.ToTensor()
        ])

    def post_seg(self, label_img, c_label_img):
        c_areas = get_all_area(c_label_img)
        centers = []
        for key, value in c_areas.items():
            num = len(value)
            if num < 10:
                continue
            centers.append(value[num//2])
        centers = set(centers)

        areas = get_all_area(label_img)
        #s_dic = {key: len(value) for key, value in areas.items()}
        #sorted_key_list = sorted(s_dic, key=lambda x: s_dic[x])
        #std_s = s_dic[sorted_key_list[len(sorted_key_list) // 2]]
        #print(s_dic, std_s)
        for key in areas:
            if len(areas[key]) < 1:
                for x, y in areas[key]:
                    label_img[x][y] = 0
            inter = set(areas[key]) & centers
            if len(inter) <= 1:
                continue
            estimator = KMeans(len(inter))
            estimator.cluster_centers_ = np.array(list(inter))
            coords = np.array(areas[key])
            estimator.fit(coords)
            max_lb = label_img.max()
            for index, class_k in enumerate(estimator.labels_):
                if class_k == 0:
                    continue
                new_lb = max_lb + class_k
                x, y = coords[index]
                label_img[x][y] = new_lb

    def __call__(self, img):
        with torch.no_grad():
            gray = unfold(bgr_to_gray(img).reshape((628,628,1)))
            #print(gray.reshape((672,672))[22:650,22:650] == bgr_to_gray(img))
            gray = Image.fromarray(gray.reshape((672,672)))
            gray_c = Image.fromarray(bgr_to_gray(img))
            gray = self.transform(gray).to(self.device)/255
            gray_c = self.transform_c(gray_c).to(self.device)
            #print(gray, gray_c)

            output = self.unet(gray.view(1, 1, self.l, self.l))
            otp = output.cpu().detach()
            prediction = np.ones_like(otp)
            prediction[otp <= 0.5] = 0
            prediction[otp > 0.5] = 1
            prediction *= 255
            prediction = prediction.reshape((self.l, self.l)).astype('uint8')
            binary_mask = prediction[22:650, 22:650]
            binary_mask = cv2.medianBlur(binary_mask, 5)
            connectivity = 4
            _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity, cv2.CV_32S)
            
            c_output = self.cnet(gray_c.view(1, 1, self.lc, self.lc))
            otp = c_output.cpu().detach()
            prediction = np.ones_like(otp)
            prediction[otp <= 0.5] = 0
            prediction[otp > 0.5] = 1
            prediction *= 255
            prediction = prediction.reshape((self.lc, self.lc)).astype('uint8')
            binary_mask = cv2.resize(prediction, (628, 628))
            binary_mask = cv2.medianBlur(binary_mask, 5)
            connectivity = 4
            _, c_label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity, cv2.CV_32S)
            self.post_seg(label_img, c_label_img)
            
        return label_img


if __name__ == "__main__":
    #segmentor = BinaryThresholding(threshold=110)
    segmentor = UnetSeg(672, 628)
    image_path = './dataset1/test/'
    result_path = './dataset1/test_RES'
    if not osp.exists(result_path):
        os.mkdir(result_path)
    image_list = sorted([osp.join(image_path, image) for image in os.listdir(image_path)])
    images = load_images(image_list)
    for index, image in enumerate(images):
        label_img = segmentor(image)
        #print(label_img[10][20])
        #x = get_all_area(label_img)
        #for key in x.keys():
        #    print(key, len(set(x[key])))
        #sys.exit(0)
        imageio.imwrite(osp.join(result_path,
         'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
