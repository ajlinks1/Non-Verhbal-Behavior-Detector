
import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics
from scipy.stats import sem
from scipy.ndimage import zoom
from skimage import io
from sklearn import datasets
import json


faces = io.imread_collection('C:\\Users\\ajank\\PycharmProjects\\Emotion_Detection\\Dataset\\Dataset')
emotions = ["content","angry","happy","surprised","flirty"]
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

class Trainer:
    def __init__(self):
        self.results = {}
        #self.images = faces.images
        self.index = 0

    def reset(self):
        print("Resetting Dataset and Previous Results .. Done")
        self.results = {}
        #self.images = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.images):
            return self.index
        else:
            while str(self.index) in self.results:
                self.index += 1
            return self.index

def extraction(gray, detected_face, offest):
    (x,y,w,h) = detected_face
    h_offset = int(offest[0]*w)
    v_offset = int(offest[1]*h)
    extracted = gray[y + v_offset:y + h, x + h_offset:x - h_offset + w]
    new_extracted = zoom(extracted, (64. / extracted.shape[0], 64. /extracted.shape[1]))
    new_extracted = new_extracted.astype(np.float32)
    new_extracted /= float(new_extracted.max())
    return new_extracted

def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %emotion) #get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect face
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10,
                                         minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face)==1:
            facefeatures = face
        elif len(face_two)==1:
            facefeatures = face_two
        elif len(face_three)==1:
            facefeatures = face_three
        elif len(face_four)==1:
            facefeatures = face_four
        else:
            facefeatures = ""

        for (x,y,w,h) in facefeatures:
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w]
            try:
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion,filenumber))
            except:
                pass
        filenumber += 1
for emotion in emotions:
    detect_faces(emotion)

def evaluate_cv(clf,X,y,K):
    cv = KFold(len(y), K, shuffle = True, random_state=0)
    scores = cross_val_score(clf, X,y,cv=cv)
    print("Scores: ", scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

def train(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set: ")
    print(clf.score(X_train,y_train))
    print("Accuracy on testing set: ")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print("Classification report: ")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.congusion_matrix(y_test,y_pred))

if __name__ == "__main__":
    svc_1 = SVC(kernel='lenear')

    trainer = Trainer()
    results = json.load(open("C:/Users/ajank/PycharmProjects/Emotion_Detection/results/results.xml"))
    trainer.results = results

    indices = [int(i) for i in trainer.results]
    data = faces.data[indices, :]

    target = [trainer.results[i] for i in trainer.results]
    target = np.array(target).astype(np.int32)

    X_train, X_test,y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state=0)

    evaluate_cv(svc_1, X_train, y_train, 5)
    train(svc_1, X_train, X_test, y_train, y_test)

    while True:
        frame = io.imread_collection('C:\\Users\\ajank\\PycharmProjects\\Emotion_Detection\\Dataset\\Dataset')
        gray, detect_faces = detect_faces()
        face_index = 0
        #cv2.putText(frame, "Press Esc to Quit" (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
        for face in detect_faces:
            (x,y,w,h) = face
            if w > 100:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
                extracted_f = extraction(gray, face, (0.3,0.05))
                frame[face_index * 64: (face_index+1)*64, -65:-1,:] = cv2.cvtColor()
                detection = detect_faces(emotion)
            face_index += 1
        cv2.imshow(face)
        if cv2.waitKey(10) & 0xFF ==27:
            break
    cv2.destroyAllWindows()


