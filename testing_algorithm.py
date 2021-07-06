import numpy as np
import pandas as pd

import urllib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import glob
import os
import random
import numpy as np
import argparse
import cv2
import logging
from time import time
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
import tensorflow.python.keras.api._v1.keras
from pathlib import Path
from tensorflow.python.keras.api._v1.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.api._v1.keras.models import Sequential, load_model, model_from_json
from tensorflow.python.keras.api._v1.keras.preprocessing import image
from tensorflow.python.keras.api._v1.keras.losses import categorical_crossentropy
from tensorflow.python.keras.api._v1.keras.optimizers import Adam
from tensorflow.python.keras.api._v1.keras.regularizers import l2
from tensorflow.python.keras.api._v1.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
import imutils


model = model_from_json(open("model.json", "r").read())
lmodel = load_model("model.h5")

emotion_map = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
img = 'C:/Users/ajank/PycharmProjects/Emotion_Detection/Presidents/000250.jpg'


    #f = f.rstrip("/").rstrip("\\")
print("Processing file: {}".format(img))

image = cv2.imread(img)
#image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)
canvas = np.zeros((220,300,3), dtype = "uint8")
frameClone = image.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
def texts(emotions, prod):
    return "{} is {:.2f}%".format(emotions, prod*100)

for(x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w, y+h), (0,255,0), 2)
    face_crop = gray[y:y+h, x:x+w]
    cropped = np.expand_dims(np.expand_dims(cv2.resize(face_crop, (48, 48)), -1), 0)

    predictions = lmodel.predict(cropped)
    max_index = int(np.argmax(predictions))
    label = emotion_map[predictions.argmax()]
    for(i, (emotion, prob)) in enumerate(zip(emotion_map, predictions)):
        #prob2 = "%.2f" % (prob*100).astype(float)
        text_1=np.vectorize(texts)
        text = text_1(emotions, prob)
        prob = prob*300
        w = prob.astype(int)

        print(text)
        #cv2.rectangle(canvas, (5, (int(i) * 35)), (int(w), (int(i) * 35)), (40, 50, 155), -1)

#        cv2.putText(canvas, text, (10, (i*35)+23),font, 0.45,(55,25,5),2)

    cv2.putText(frameClone,label,(x,y-10),font, 0.45, (55,25,5),2)
    #cv2.rectangle(frameClone,(x,y),(x+20, y-60), (140,50,155),2)
    cv2.putText(gray, emotion_map[max_index], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('frame', gray)
#cv2.imshow("Face", frameClone)
#cv2.imshow("Probabilities", canvas)


cv2.waitKey(0)
cv2.destroyAllWindows()
