import cv2
import imageio
import numpy as np
import os
import os.path as osp


def visual(img, gt):
    img = cv2.imread(img, -1)
    gt = cv2.imread(gt, -1)
    label = np.unique(gt)
    height, width = img.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[gt==lab, :] = color
    return img.astype(np.uint8), visual_img.astype(np.uint8)


if __name__ == "__main__":
    # train->origin & gt; test->origin & pred
    mode = "train"
    index = "023"
    img = "./dataset1/" + mode + "/t" + index + ".tif"
    if mode == "train":
        gt = "./dataset1/train_GT/SEG/man_seg" + index + ".tif"
        pred = "./dataset1/train_RES_UNET/mask" + index + ".tif"
        pred_post = "./dataset1/train_RES/mask" + index + ".tif"
    else:
        gt = None
        pred = "./dataset1/test_RES_UNET/mask" + index + ".tif"
        pred_post = "./dataset1/test_RES/mask" + index + ".tif"
    cell, cell_pred = visual(img, pred)
    _, cell_pred_post = visual(img, pred_post)
    if gt is not None:
        _, cell_gt = visual(img, gt)
    #print(cell.shape, cell_mask.shape)
    cv2.namedWindow("cell", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cell", 400, 400)
    cv2.moveWindow("cell", 100, 100)
    cv2.imshow("cell", cell)
    cv2.namedWindow("cell_pred", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cell_pred", 400, 400)
    cv2.moveWindow("cell_pred", 550, 100)
    cv2.imshow("cell_pred", cell_pred)
    cv2.namedWindow("cell_pred_post", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cell_pred_post", 400, 400)
    cv2.moveWindow("cell_pred_post", 550, 550)
    cv2.imshow("cell_pred_post", cell_pred_post)
    if gt is not None:
        cv2.namedWindow("cell_gt", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cell_gt", 400, 400)
        cv2.moveWindow("cell_gt", 1000, 100)
        cv2.imshow("cell_gt", cell_gt)
    cv2.waitKey(0)
