from __future__ import absolute_import

import cv2
from PIL import Image
import imageio
import numpy as np
import os, sys
import os.path as osp
import models, models2
import torch
from sklearn.cluster import KMeans
from torchvision import transforms
from jaccard import get_all_area


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
        pass

    def post_seg(self, label_img, c_label_img):
        c_areas = get_all_area(c_label_img)
        centers = []
        for key, value in c_areas.items():
            num = len(value)
            if num < 10:
                for x, y in value:
                    c_label_img[x][y] = 0
            centers.append(value[num//2])
        centers = set(centers)

        areas = get_all_area(label_img[:,:,0])
        for key in areas:
            if len(areas[key]) < 50:
                for x, y in areas[key]:
                    label_img[x][y] = 0
            inter = set(areas[key]) & centers
            if len(inter) == 0:
                max_clb = c_label_img.max() + 1
                for x, y in areas[key]:
                    c_label_img[x][y] = max_clb

        return cv2.watershed(label_img, c_label_img)
        #cv2.namedWindow("cell", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("cell", 400, 400)
        #cv2.moveWindow("cell", 100, 100)
        #cv2.imshow("cell", c_label_img.astype(np.uint16))
        #cv2.waitKey(0)

    def draw(self, img):
        drawing = False # true if mouse is pressed
        mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        ix,iy = -1,-1

        # mouse callback function
        def draw_circle(event,x,y,flags,param):
            global ix,iy,drawing,mode

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix,iy = x,y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    cv2.circle(visual_img,(x,y),3,(255,255,255),-1)
                    cv2.circle(target,(x,y),3,(255,255,255),-1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.circle(visual_img,(x,y),3,(255,255,255),-1)
                cv2.circle(target,(x,y),3,(255,255,255),-1)

        target = np.zeros(img.shape, np.uint8)
        visual_img = np.zeros(img.shape, np.uint8)

        label = np.unique(img)
        for lab in label:
            if lab == 0:
                continue
            color = np.random.randint(low=0, high=255, size=3)
            visual_img[img[:,:,0]==lab, :] = color

        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)

        while(1):
            cv2.imshow('image',visual_img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == ord('q'):
                return 0
            elif k == 27:
                break

        cv2.destroyAllWindows()
        return target[:,:,0]


    def __call__(self, img):
        img[img>0] = 255
        binary_mask = img
        label_img = binary_mask.reshape((628, 628, 1))
        label_img = np.concatenate([label_img, label_img, label_img], axis=2)
            
        binary_mask = self.draw(label_img)
        if type(binary_mask) == int:
            return 0
        _, c_label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_32S)

        label_img = self.post_seg(label_img, c_label_img)
        label_img[label_img==-1] = 0
        for i in range(1, label_img.max()+1):
            if np.sum(label_img == i)>10000:
                label_img[label_img==i] = 0
            
        return label_img


if __name__ == "__main__":
    #segmentor = BinaryThresholding(threshold=110)
    segmentor = UnetSeg(628, 628)
    image_path = './dataset1/test_RES_UNET/'
    result_path = './dataset1/test_RES'
    if not osp.exists(result_path):
        os.mkdir(result_path)
    image_list = sorted([osp.join(image_path, image) for image in os.listdir(image_path)])
    images = load_images(image_list)
    for index, image in enumerate(images):
        index = 23
        image = images[index]
        label_img = segmentor(image[:,:,0])
        if type(label_img) == int:
            sys.exit(0)
        #x = get_all_area(label_img)
        #for key in x.keys():
        #    print(key, len(set(x[key])))
        imageio.imwrite(osp.join(result_path,
         'mask{:0>3d}.tif'.format(index)), label_img.astype(np.uint16))
        sys.exit(0)
