import cv2
from core.opencv_face import face_eye_detector, IOU
from core.dlib_face import frontal_face_detector


# 视频中的人脸类
class Face():
    def __init__(self):
        self.id = None
        # 此帧中该脸是否为正脸
        self.frontal = False
        # 此帧中该脸矩形框的位置
        self.position = None
        # 该脸的人脸特征（只要该脸出现过正脸即可得到）
        self.shape = None
        # 该脸的状态（-1为不在，0为侧脸，1、2为不同专注度的正脸，测试阶段只有1）
        self.station = None
        # 目前所有帧该脸的状态列表
        self.time = []

    # 获取脸的状态
    def station_analyze(self):
        if self.position is None:
            return -1
        ############################
        elif self.frontal is True:
            return 1
        else:
            return 0

    # 对比该脸和newface，是否是同一个人
    def compare(self, newface):
        if self.shape is None or newface.shape is None:
            return False
        else:
            ###############################
            if self.shape == newface.shape:
                return True
            else:
                return False


# 先用dlib找正脸，存储正脸后将正脸的矩形涂黑（避免前后找重复），再用opencv找正脸和侧脸
def face_detector(image):
    output = []
    for dic in frontal_face_detector(image):
        newone = Face()
        newone.frontal = True
        newone.position = dic['position']
        newone.shape = dic['shape']
        newone.station = newone.station_analyze()
        output.append(newone)
        x, y, w, h = newone.position
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), thickness=-1)
    for dic in face_eye_detector(image):
        newone = Face()
        newone.frontal = False
        newone.position = dic['face']
        newone.station = newone.station_analyze()
        output.append(newone)
    return output


# 第second帧的人脸检测结果为newface_list，目前所有人的人脸类存于allface_dic，对其进行更新
def substitute(allface_dic, newface_list, second):
    # 存储allface_dic在此帧中发生过更新的脸的id
    Is_refresh = {}
    for newface in newface_list:
        # 该脸与前一帧的脸是否有较大面积重合
        ok = False
        for id in allface_dic:
            value = allface_dic[id].position
            # 人脸框为空说明已经前一帧已离开，暂设这一帧仍然离开
            if value is None:
                allface_dic[id].time += [-1]
                continue
            # 若该脸与编号为id的前一帧的脸有较大重合
            elif IOU(newface.position, value) > 0.5:
                # 排除该脸是错误测出的小框，只有面积相当且有较大重合才说明是前后帧中的同一张脸
                if newface.position[2]*newface.position[3]/(value[2]*value[3]) > 0.6:
                    # 添加此帧的状态
                    temp_time = allface_dic[id].time + [newface.station]
                    # 若该脸为正脸，则更新shape，否则保持之前的shape
                    temp_shape = allface_dic[id].shape
                    # 其他的属性全部与该脸相同
                    allface_dic[id] = newface
                    allface_dic[id].time = temp_time
                    if newface.shape is not None:
                        allface_dic[id].shape = newface.shape
                    else:
                        allface_dic[id].shape = temp_shape
                    Is_refresh[id] = True
                # 有较大重合但面积相差悬殊，排除该脸
                else:
                    pass
                # 该脸已分析过，不再与其他前一帧的脸比对面积重合度（若与多个前一帧的脸都有较大重合，则需要修改）
                ok = True
                break
        # 该脸已分析过，直接开始下一张脸
        if ok is True:
            continue
        # 该脸与所有前一帧的脸都进行比对，没有与其重合较大的，故认为该脸为新进入场景中的脸
        else:
            # 遍历前一帧的脸，事实上只需要遍历前一帧不在（position为空）的脸
            for id in allface_dic:
                # 用shape对比人脸相似度（只有该脸为正脸且编号为id的脸出现过正脸，才需要比较），若为同一个人
                if allface_dic[id].compare(newface) is True:
                    # 该脸已分析过，不再与其他前一帧的脸对比
                    ok = True
                    # 该脸即为之前离开，现在又重新回到镜头的编号为id的人脸
                    Is_refresh[id] = True
                    allface_dic[id].position = newface.position
                    allface_dic[id].shape = newface.shape
                    # 之间暂设离开，在time中写入了离开状态，现在需要修改回来
                    allface_dic[id].time[-1] = newface.station_analyze()
                    break
            # 该脸为已出现过的脸，不需要新建id
            if ok is True:
                continue
            # 该脸未出现过（或者出现过，但前后没有均出现过正脸），新建一个新脸
            else:
                allface_dic[str(len(allface_dic))] = newface
                # 新建的脸在之间的帧中都处于离开状态，故添上（second-1）个离开状态
                allface_dic[str(len(allface_dic) - 1)].time = [-1]*(second-1)+[newface.station_analyze()]
                Is_refresh[str(len(allface_dic) - 1)] = True
    for id in allface_dic:
        # Is_refresh中没有的id说明编号为id的脸在新的一帧中找不到对应脸，则认为已离开
        if id not in Is_refresh.keys():
            allface_dic[id].position = None
