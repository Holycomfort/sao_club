import cv2
import imageio
import numpy as np
import os
import os.path as osp
import queue


# 连通域用含有所有内点坐标的列表来表示，坐标用(行，列)来表示
def Intersection(area1, area2):
    return len(set(area1) & set(area2))

def Union(area1, area2):
    return len(set(area1) | set(area2))

def Jaccard(area1, area2):
    return Intersection(area1, area2) / Union(area1, area2)

# 得到字典，key从0开始，value为area
def get_all_area(mask):
    areas = {}
    num = 0
    discover = [[False for i in range(628)] for i in range(628)]
    q = queue.Queue()
    for x in range(628):
        for y in range(628):
            if discover[x][y] == True:
                continue
            else:
                discover[x][y] = True
                if mask[x][y] == 0 or (x in [0, 627]) or (y in [0, 627]):
                    continue
                else:
                    areas[num] = []
                    q.put((x, y))
                    while not q.empty():
                        item = q.get()
                        areas[num].append(item)
                        if 0 < item[0] < 627 and 0 < item[1] < 627:
                            if discover[item[0] - 1][item[1]] == False and mask[item[0] - 1][item[1]] != 0:
                                q.put((item[0] - 1, item[1]))
                                discover[item[0] - 1][item[1]] = True
                            if discover[item[0] + 1][item[1]] == False and mask[item[0] + 1][item[1]] != 0:
                                q.put((item[0] + 1, item[1]))
                                discover[item[0] + 1][item[1]] = True
                            if discover[item[0]][item[1] - 1] == False and mask[item[0]][item[1] - 1] != 0:
                                q.put((item[0], item[1] - 1))
                                discover[item[0]][item[1] - 1] = True
                            if discover[item[0]][item[1] + 1] == False and mask[item[0]][item[1] + 1] != 0:
                                q.put((item[0], item[1] + 1))
                                discover[item[0]][item[1] + 1] = True
                    num += 1
    return areas

def AOPCN(pred_areas, gt_areas):
    jac = {}
    for key_p in pred_areas.keys():
        for key_g in gt_areas.keys():
            area_p, area_g = pred_areas[key_p], gt_areas[key_g]
            if Intersection(area_p, area_g) > 0.5 * len(area_g):
                jac[key_p] = Jaccard(area_g, area_p)
                gt_areas.pop(key_g)
                break
        if key_p not in jac.keys():
            jac[key_p] = 0
    return jac


if __name__ == "__main__":
    index = "006"
    pred = "./dataset1/train_RES_UNET/mask" + index + ".tif"
    gt = "./dataset1/train_GT/SEG/man_seg" + index + ".tif"
    pred = cv2.imread(pred, -1)
    gt = cv2.imread(gt, -1)
    pred_areas = get_all_area(pred)
    gt_areas = get_all_area(gt)
    print(AOPCN(pred_areas, gt_areas))
