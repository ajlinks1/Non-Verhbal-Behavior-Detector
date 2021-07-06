import sys
import os
import numpy as np
from os import walk
import cv2

imdir = "C:/Users/lbrooks/PycharmProjects/test/google-images-download/images/Dataset"

n = 249

jpg_files = set()
xml_files = set()

for imfile in os.listdir(imdir):
    name, ext = imfile.split('.')
    if ext == 'jpg':
        jpg_files.add(name)
    elif ext == 'xml':
        xml_files.add(name)

for imfile in jpg_files.intersection(xml_files):
    os.rename(os.path.join(imdir,imfile+'.jpg'),os.path.join(imdir, str(n)+'.jpg'))
    os.rename(os.path.join(imdir, imfile+'.xml'),os.path.join(imdir, str(n)+'.xml'))
    n += 1

for imfile in os.scandir(imdir):
    os.rename(imfile.path, os.path.join(imdir, '{:06}.jpg'.format(n)))
    n += 1
# width to resize
width = int(640)
# height to resize
height = int(480)
# location of the input dataset
input_dir = ("C:/Users/lbrooks/PycharmProjects/test/google-images-download/images/Dataset")
# location of the output dataset
out_dir = ("C:/Users/lbrooks/PycharmProjects/test/google-images-download/images/Dataset")

print("Working...")

# get all the pictures in directory
images = []
ext = (".jpeg", ".jpg", ".png")

for (dirpath, dirnames, filenames) in walk(input_dir):
    for filename in filenames:
        if filename.endswith(ext):
            images.append(os.path.join(dirpath, filename))

for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[:2]
    pad_bottom, pad_right = 0, 0
    ratio = w / h

    if h > height or w > width:
        # shrinking image algorithm
        interp = cv2.INTER_AREA
    else:
        # stretching image algorithm
        interp = cv2.INTER_CUBIC

    w = width
    h = round(w / ratio)
    if h > height:
        h = height
    w = round(h * ratio)
    pad_bottom = abs(height - h)
    pad_right = abs(width - w)

    scaled_img = cv2.resize(img, (w, h), interpolation=interp)
    padded_img = cv2.copyMakeBorder(
        scaled_img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(os.path.join(out_dir, os.path.basename(image)), padded_img)

print("Completed!")
