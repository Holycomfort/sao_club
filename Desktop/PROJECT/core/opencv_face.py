import cv2
import numpy as np


# opencv自带分类器，若效果不佳可以考虑自己训练
# 侧脸分类器
pface_detector = cv2.CascadeClassifier("static/alib/haarcascade_profileface.xml")
# 正脸分类器，用于dlib漏掉的正脸，如低头族
fface_detector = cv2.CascadeClassifier("static/alib/haarcascade_frontalface_default.xml")
# 眼睛分类器，目前无用处
eye_detector = cv2.CascadeClassifier("static/alib/haarcascade_eye.xml")


# 计算两矩形重合面积占较小矩形的比例
def IOU(f1, f2):
    x1, y1, w1, h1 = f1
    x2, y2, w2, h2 = f2
    s1 = f1[2]*f1[3]
    s2 = f2[2]*f2[3]
    xend = max(x1+w1, x2+w2)
    xstart = min(x1, x2)
    yend = max(y1+h1, y2+h2)
    ystart = min(y1, y2)
    dw = w1+w2-(xend-xstart)
    dh = h1+h2-(yend-ystart)
    if dw < 0 or dh < 0:
        return 0
    else:
        return dw*dh/min(s1, s2)


# 筛掉某些错误分类的人脸框，可以根据大小、颜色来进行
##################
def reject(roi):
    return False


# 对于找出的人脸框，先判断是否筛掉，再看与之前找出的框重合程度如何，如重合程度大，取面积大者替换之前的框的位置，否则添加新框
def addface(outlist, newone, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = newone['face']
    eye = newone['eye']
    if reject(gray[face[1]: face[1]+face[3], face[0]: face[0]+face[2]]) is True:
        return
    else:
        for i, dic in enumerate(outlist):
            f = dic['face']
            iou = IOU(f, face)
            if iou < 0.5:
                continue
            else:
                if f[2]*f[3] < face[2]*face[3]:
                    outlist[i]['face'] = face
                    outlist[i]['eye'] = eye
                else:
                    return
        outlist.append(newone)


# 找出所有的正脸和侧脸（侧脸只能找一边，另一侧需要对原图做镜像处理），筛选后放入output中
def face_eye_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ffaces = fface_detector.detectMultiScale(gray, 1.1, 2)
    pfaces1 = pface_detector.detectMultiScale(gray, 1.1, 2)
    try:
        faces = np.vstack((ffaces, pfaces1))
    except:
        if len(ffaces) is 0:
            faces = pfaces1
        else:
            faces = ffaces
    # 输出为列表，列表每项为一个字典，key分别为face和eye，face的value为元组（xywh），eye的value为元组列表（xyzw）
    output = []
    if len(faces) is 2:
        return []
    for x, y, w, h in faces:
        dic = {}
        dic['face'] = (x, y, w, h)
        dic['eye'] = []
        roi_gray = gray[y: y+h, x: x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            dic['eye'].append((x+ex, y+ey, ew, eh))
        addface(output, dic, image)
    gray = cv2.flip(gray, 1)
    pfaces2 = pface_detector.detectMultiScale(gray, 1.1, 2)
    width = gray.shape[1]
    for x, y, w, h in pfaces2:
        dic = {}
        dic['face'] = (width-x-w, y, w, h)
        dic['eye'] = []
        roi_gray = gray[y: y+h, x: x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            dic['eye'].append((width-(x+ex)-ew, y+ey, ew, eh))
        addface(output, dic, image)
    return output


if __name__ == '__main__':
    image = cv2.imread("/Users/apple/Desktop/1.jpeg")
    print(image.shape)
    res = face_eye_detector(image)
    print(res)
    for dic in res:
        f = dic['face']
        cv2.rectangle(image, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 2)
        for e in dic['eye']:
            cv2.rectangle(image, (e[0], e[1]), (e[0] + e[2], e[1] + e[3]), (0, 255, 0), 2)
    cv2.imshow('', image)
    cv2.waitKey(0)
