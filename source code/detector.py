import os
import numpy as np
import cv2

import pickle
face_mask = ['No mask', 'Masked']


def LoR_create_Xb(X):
    N = X.shape[0]
    ones = np.ones([N, 1])
    Xb = np.hstack([ones, X])
    return Xb


def predict(X, W):  # LoR_find_Yhat_mul_class
    Xb = LoR_create_Xb(X)
    Z = np.dot(Xb, W)
    Yhat = np.exp(Z)/np.exp(Z).sum(axis=1, keepdims=True)
    return Yhat


# Load face detection and face mask model
with open('outfile', 'rb') as fp:
    myW = pickle.load(fp)

faceNet = cv2.dnn.readNet(os.path.join(r'deploy.prototxt.txt'),
                          os.path.join('res10_300x300_ssd_iter_140000.caffemodel'))


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(9))
frame_height = int(cap.get(16))


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    image_train = np.empty((0, 15552), int)  # 5184

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (54, 96))
        image = np.array(face)
        image = image.reshape(1, -1)
        image_train = np.vstack((image_train, image))/255.0
        result = np.argmax(predict(image, myW))

        if result == 0:
            label = face_mask[result]
            color = (0, 0, 255)
        else:
            label = face_mask[result]
            color = (0, 255, 0)

        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1600, 900)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
