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

from tensorflow.python.keras.api._v1.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.api._v1.keras.models import Sequential, load_model
from tensorflow.python.keras.api._v1.keras.losses import categorical_crossentropy
from tensorflow.python.keras.api._v1.keras.optimizers import Adam
from tensorflow.python.keras.api._v1.keras.regularizers import l2
from tensorflow.python.keras.api._v1.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

emotion_map = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
model = load_model('C:/Users/ajanks/PycharmProjects/Emotion_Detection/keras_model/model_5-49-0.62.hdf5')
#model.load_weights('model_5-49-0.62.hdf5')
img = 'C:/Users/ajanks/PycharmProjects/Emotion_Detection/Presidents/000250.jpg'

print("Processing file: {}".format(img))
image = cv2.imread(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1)
for(x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w, y+h), (0,255,0), 1)
    face_crop = image[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop,(48,48))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop = face_crop.astype('float32')/255
    face_crop = np.asarray(face_crop)
    face_crop = face_crop.reshape(1,1,face_crop.shape[0], face_crop.shape[1])
    predictions = model.predict(face_crop)
    result = np.argmax(predictions[0])
    emotion_prediction = emotion_map[result]
    cv2.putText(image, emotion_prediction, (x,y), font, 1, (200,0,0),3,cv2.LINE_AA)
cv2.imshow('result', image)
cv2.imwrite('result_emotion_detection.jpg', image)
cv2.waitKey(0)
