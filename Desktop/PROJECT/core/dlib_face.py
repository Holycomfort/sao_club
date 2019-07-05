import cv2
import dlib
import sys


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("static/alib/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("static/alib/dlib_face_recognition_resnet_model_v1.dat")


def frontal_face_detector(image):
    # 输出为列表，每个元素为一个字典，key分别为position和shape，position对应的value为（xywh）元组
    output = []
    # 视频中的最后一帧（虚拟帧，不具有物理意义）会引起报错
    try:
        dets = detector(image, 1)
    except:
        return output
    for k, d in enumerate(dets):
        dic = {}
        width = d.right() - d.left()
        height = d.bottom() - d.top()
        dic['position'] = (d.left(), d.top(), width, height)
        # 利用预测器预测人脸特征点
        shape = predictor(image, d)
        dic['shape'] = shape
        output.append(dic)
    return output


if __name__ == '__main__':
    img = cv2.imread('/Users/apple/Desktop/1.jpeg')
    res = frontal_face_detector(img)
    for dic in res:
        shape = dic['shape']
        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 4, (0, 255, 0), -1, 8)
            cv2.putText(img, None, (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        # 显示一下处理的图片，然后销毁窗口
        cv2.imshow('face', img)
        cv2.waitKey(0)
