import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
django.setup()
import cv2
from core.face import face_detector, substitute


allface_dic = {}
fm = 1
while True:
    try:
        frame = cv2.imread('../static/figure/rawphoto/sss/'+str(fm)+'.jpg')
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
                cv2.rectangle(frame, (face[0], face[1]),
                              (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
        cv2.imshow('', frame)
        cv2.waitKey(0)
        fm += 1
    except Exception as e:
        print(repr(e))
        break


dic = {tid:tface.time for tid, tface in allface_dic}