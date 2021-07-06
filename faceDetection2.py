import dlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# from cnn_face_detector import rects
# from getImages import image
import urllib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# change to CNN but for prototype just do k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import glob
import os
import random

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import logging
from time import time
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
fishface = cv2.createFisherFaceRecognizer()

image = 'C:/Users/ajanks/PycharmProjects/Emotion_Detection/Image'
frontalface_detector = dlib.get_frontal_face_detector()

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

for f in glob.glob(os.path.join(image, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    # dlib.hit_enter_to_continue()

if (len(sys.argv[1:]) > 0):
    for f in glob.glob(os.path.join(image, "*.jpg")):
        img = dlib.load_rgb_image(f)
        dets, scores, idx = detector.run(img, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}, score: {}, face_type:{}".format(
                d, scores[i], idx[i]))

landmark_predictions = dlib.shape_predictor('C:/Users/ajanks/PycharmProjects/Emotion_Detection/shape_predictor_68_face_landmarks.dat')


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.left() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def get_landmarks(image):
    for f in glob.glob(os.path.join(image, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = landmark_predictions(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            landmarks = []
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                landmarks.append(x)
                landmarks.append(y)

            cv2.imshow("Output", img)
    return landmarks
        # dlib.hit_enter_to_continue()
        # cv2.waitKey(0)

emotion_map = {"0": "Angry", "1": "Happy", "2": "Sad", "3": "Surprise", "4": "Neutral"}
df = pd.read_csv("C:/Users/ajanks/PycharmProjects/Emotion_Detection/Data/fer2013.csv")
df.head()

# num_row = 0
# for row in open("C:/Users/ajanks/PycharmProjects/Emotion_Detection/Data/fer2013.csv"):
#     num_row += 1
#
# X = df['pixels'].values
# Y = df['emotion'].values
#
# print("Total dataset size:")
# print("pixels", X)
# print("emotion", Y)
# print("number of rows", num_row)

clahe = cv2.createCLAHE(cliplimit=2.0, tileGridSize=(8,8))
svm_clf2 = SVC(kernel='linear', probability=True, tol=1e-3)

def get_files(emotion):
    m_path = 'C:/Users/ajanks/PycharmProjects/Emotion_Detection/Images'
    i_path = glob.glob(m_path+emotion+'/*')
    random.shuffle(i_path)
    training = image[:int(len(i_path) * 0.8)]
    testing = image[-int(len(i_path) * 0.2)]
    return training, testing


def make_sets():
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []

    for emo in emotion_map:
        training, testing = get_files(emo)
        for item in training:
            images = cv2.imread(item)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            clahe_gray = clahe.apply(gray)
            landmarks = get_landmarks(clahe_gray)
            training_data.append(landmarks)
            training_labels.append(emotion_map.index(emo))
        for item in testing:
            images = cv2.imread(item)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            clahe_gray = clahe.apply(gray)
            landmarks = get_landmarks(clahe_gray)
            testing_data.append(landmarks)
            testing_labels.append(emotion_map.index(emo))
    return training_data, training_labels, testing_data, testing_labels


def run_recognizer():
    training_data, training_labels, testing_data, testing_labels = make_sets()
    print("training classifier")
    print("size of training set is: ", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in testing_data:
        test, conf = fishface.predict(image)
        if test == testing_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect))


meta_score = []
for i in range(0, 10):
    correct = run_recognizer()
    print("got ", correct, " percent correct!")
    meta_score.append(correct)
print("\n\nend score: ", np.mean(meta_score), " percent correct!")


