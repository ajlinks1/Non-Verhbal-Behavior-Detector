import numpy as np
import pandas as pd
import os
from PIL import Image
# put the directory where the csv file is located
df = pd.read_csv('/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Data/fer2013.csv')
#___________________________________________________________________________________
# read what the csv column emotion is set to 
#__________________________________________________________________________________
df0 = df[df['emotion'] == 0]
df1 = df[df['emotion'] == 1]
df2 = df[df['emotion'] == 2]
df3 = df[df['emotion'] == 3]
df4 = df[df['emotion'] == 4]
df5 = df[df['emotion'] == 5]
df6 = df[df['emotion'] == 6]

#_____________________________________________________________________________________________
# creates a directory for each emotion -change to the where the other dataset info is located
# I manually created the folders but this code does work to automatically make the folders
#_____________________________________________________________________________________________

# os.mkdir("/Users/ajank/PycharmProjects/Jupyter/Project3_data/Angry/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Disgust/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Fear/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Happy/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Sad/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Surprise/")
# os.mkdir("/Users/blakemyers/Desktop/Jupyter/Project3_data/Neutral/")

d=0
for image_pixels in df0.iloc[1:,1]:
    image_string = image_pixels.split(' ') #pixels in the csv is separated by spaces / splits up the pixels with ' '
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48) #resizes the image
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Angry/img_%d.jpg"%d, "JPEG")   #saves the image to the emotion specific folder as img_number)
    d+=1

d=0
for image_pixels in df1.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Disgust/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df2.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Fear/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df3.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Happy/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df4.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Sad/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df5.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Surprise/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df6.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("/Users/lbrooks/PycharmProjects/NonVerbalBehaviorDetection1/Neutral/img_%d.jpg"%d, "JPEG")
    d+=1
