
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import cv2

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

#from imutils import face_utils
import numpy as np
import argparse
#import imutils
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
from tensorflow.python.keras.api._v1.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import brewer2mpl
from keras import layers
from keras import models
from keras import optimizers


image = 'C:/Users/ajank/PycharmProjects/Emotion_Detection/Images'
sys.path.insert(0, image)
os.chdir(image)
Model_Path='C:/Users/ajank/PycharmProjects/Emotion_Detection/model.h5'
#os.mkdir(Model_Path)

num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48

data = pd.read_csv('C:/Users/ajank/PycharmProjects/Emotion_Detection/Data/fer2013.csv')
data.tail()

def emotion_count(y_train, classes):
    """
    The function re-classify picture with disgust label into angry label
    """
    emo_classcount = {}
    print('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount


def load_data(sample_split=0.3, usage='Training', classes=['Angry', 'Happy'], filepath='C:/Users/ajank/PycharmProjects/Emotion_Detection/Data/fer2013.csv'):
    """
    The function load provided CSV dataset and further reshape, rescale the data for feeding
    """
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data) * sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X = []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X.append(each_pixel)
    ## reshape into 48*48*1 and rescale
    X = np.array(X)
    X = X.reshape(X.shape[0], 48, 48, 1)
    X = X.astype("float32")
    X /= 255

    y_train, new_dict = emotion_count(data.emotion, classes)
    y_train = to_categorical(y_train)
    return X, y_train

emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

X_test, y_test = load_data(sample_split=1.0,classes=emo,
usage='PrivateTest')

X_train, y_train = load_data(sample_split=1.0,classes=emo,
usage= 'Training')

X_val,y_val = load_data(sample_split=1.0,classes=emo,
usage= 'PublicTest')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

def save_data(X_test, y_test, fname=''):
    """
    The function stores loaded data into numpy form for further processing
    """
    np.save( 'X_test' + fname, X_test)
    np.save( 'y_test' + fname, y_test)
save_data(X_test, y_test,"_privatetest6_100pct")
X_fname = 'X_test_privatetest6_100pct.npy'
y_fname = 'y_test_privatetest6_100pct.npy'
X = np.load(X_fname)
y = np.load(y_fname)
print ('Private test set')
y_labels = [np.argmax(lst) for lst in y]
counts = np.bincount(y_labels)
labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
print (zip(labels, counts))

def overview(start, end, X):
    """
    The function is used to plot first several pictures for overviewing inputs format
    """
    fig = plt.figure(figsize=(20,20))
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(16,12,i+1)
        ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()
overview(0,191, X)

input_img = X[6:7,:,:,:]
print (input_img.shape)
plt.imshow(input_img[0,:,:,0], cmap='gray')
plt.show()

y_train = y_train
y_public = y_val
y_private = y_test
y_train_labels  = [np.argmax(lst) for lst in y_train]
y_public_labels = [np.argmax(lst) for lst in y_public]
y_private_labels = [np.argmax(lst) for lst in y_private]


def plot_distribution(y1, y2, data_names, ylims=[1000, 1000]):
    """
    The function is used to plot the distribution of the labels of provided dataset
    """
    colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(np.arange(1, 7), np.bincount(y1), color=colorset, alpha=0.8)
    ax1.set_xticks(np.arange(1.25, 7.25, 1))
    ax1.set_xticklabels(labels, rotation=60, fontsize=14)
    ax1.set_xlim([0, 8])
    ax1.set_ylim([0, ylims[0]])
    ax1.set_title(data_names[0])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(1, 7), np.bincount(y2), color=colorset, alpha=0.8)
    ax2.set_xticks(np.arange(1.25, 7.24, 1))
    ax2.set_xticklabels(labels, rotation=60, fontsize=14)
    ax2.set_xlim([0, 8])
    ax2.set_ylim([0, ylims[1]])
    ax2.set_title(data_names[1])
    plt.tight_layout()
    plt.show()


plot_distribution(y_train_labels, y_public_labels,
                  ['Train dataset', 'Public dataset'],
                  ylims=[8000, 1000])

pixels = data['pixels'].tolist()

faces = []
for pixel_sequence in pixels:
    face = [int(pixels) for pixels in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width,height)
    #face = face/255.0
    #face = cv2.resize(face.astype('uint8'), (width,height))
    faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data['emotion']).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.summary()

model.compile(loss = categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9,
                                                               beta_2=0.99, epsilon=1e-7), metrics=['accuracy'])
lr_reducer=ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
tensorboard=TensorBoard(log_dir='./logs')
early_stopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

checkpointer=ModelCheckpoint(Model_Path, monitor='val_loss', verbose=1, save_best_only=True)

history = model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(np.array(X_test),np.array(y_test)),
                    shuffle=True, callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])

scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))
model.save(Model_Path)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')
model.save('model.h5')
print("Saved model to disk")
loaded = load_model("model.h5")
print("loaded model:", loaded)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
