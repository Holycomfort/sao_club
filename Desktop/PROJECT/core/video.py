import cv2
from core.face import face_detector, substitute


# 目前出现过的所有脸的字典，key为id，value为Face类存储的信息
allface_dic = {}

cap = cv2.VideoCapture('/Users/apple/Desktop/3.mov')
fm = 1
# 隔20帧取一帧
timeF = 20

if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

while rval:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1+(fm-1)*timeF)
    rval, frame = cap.read()
    outlist = face_detector(frame)
    if fm is 1:
        for i, f in enumerate(outlist):
            allface_dic[str(i)] = f
            allface_dic[str(i)].time = [f.station_analyze()]
    else:
        substitute(allface_dic, outlist, fm)
    print([(id, face.position, face.frontal) for id, face in allface_dic.items()])

    for id in allface_dic:
        face = allface_dic[id].position
        if face is not None:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
    cv2.imshow('', frame)
    cv2.waitKey(0)
    fm += 1


cap.release()
