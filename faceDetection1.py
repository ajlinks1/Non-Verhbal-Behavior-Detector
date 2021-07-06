import argparse
import urllib
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

from dlib.python_examples.cnn_face_detector import rects
from imutils import face_utils

from getImages import image

frontalface_detector = dlib.get_frontal_face_detector()

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
predictor = dlib.shape_predictor(args["shape_predictor"])


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def detect_face(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        rects = frontalface_detector(image, 1)
        if len(rects) < 1:
            return "No Face Detected"
    finally:
        return "Face detected successfully"


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image, interpolation='nearest')

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

plt.axis('off')
plt.show()
