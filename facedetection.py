import dlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

#from cnn_face_detector import rects
#from getImages import image
import urllib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#change to CNN but for prototype just do k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import glob
import os

image = 'C:/Users/lbrooks/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/dlib/examples/faces'
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
    dlib.hit_enter_to_continue()

if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image('2007_007763.jpg')
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

landmark_predictions = dlib.shape_predictor('C:/Users/ajanks/PycharmProjects/Emotion_Detection/shape_predictor_68_face_landmarks.dat')

def get_landmarks(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
    except Exception as e:
        print("Please check the URL and try again.")
        return None,None
    faces = frontalface_detector(image, 1)

    if len(faces):
        landmarks = [(p.x, p.y) for p in landmark_predictions(image, faces[0]).parts()]
    else:
        return None, None
    return image, landmarks

def image_landmarks(image, face_landmarks):
    radius = -1
    circle_thick = 4
    image_copy = image.copy()
    for (x,y) in face_landmarks:
        cv2.circle(image_copy, (x,y), circle_thick, (255,0,0), radius)
        plt.imshow(image_copy, interpolation='nearest')
        plt.axis('off')
        plt.show()

emotion_map = {"0":"Angry", "1":"Happy", "2":"Sad", "3":"Surprise", "4":"Neutral"}
df = pd.read_csv("./ferdata.csv")
df.head()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataX, dataY, test_size=0.1,
                                                random_state=42, stratify=dataY)

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, Ytrain)
predictions = knn.predict(Xtest)

print(accuracy_score(Ytest, predictions)*100)