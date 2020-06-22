import cv2
import numpy as np
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

samples = []
labels = []

path = r"C:\path\to\file\folder"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+"/" +item):
            im = Image.open(path+"/"+item)
            f, e = os.path.splitext(path+"/"+item)
            imResize = im.resize((70,70), Image.ANTIALIAS)
            imResize.save(f + ' resized.png', 'png', quality=90)

#resize()

def deskew():
    for item in dirs:
        if os.path.isfile(path+"/" +item):
            im = Image.open(path+"/"+item)
            #gray = im.convert('L')  # convert the image into single channel i.e. RGB to grayscale
            gray = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2GRAY)

            img = np.array(gray)
            #print(img.dtype)
            f, e = os.path.splitext(path+"/"+item)
            m = cv2.moments(gray)
            if abs(m['mu02']) < 1e-2:
                return img.copy()
            skew = 1 * m['mu11'] / m['mu02']
            M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            plt.imsave(f+'deskewed.png', img)

#deskew()

def flip():
    for item in dirs:
        if os.path.isfile(path+"/" +item):
            im = Image.open(path+"/"+item)
            f, e = os.path.splitext(path+"/"+item)
            imFlip = ImageOps.flip(im)
            imFlip.save(f + ' flipped.png', 'png', quality=90)

#flip()

def mirror():
    for item in dirs:
        if os.path.isfile(path+"/" +item):
            im = Image.open(path+"/"+item)
            f, e = os.path.splitext(path+"/"+item)
            imMir = ImageOps.mirror(im)
            imMir.save(f + ' mirror.png', 'png', quality=90)

#mirror()

def crop_resize():
    for item in dirs:
        if os.path.isfile(path+"/" +item):
            im = Image.open(path+"/"+item)

            f, e = os.path.splitext(path + "/" + item)
            nx, ny = im.size  # Get dimensions

            size = 14 #square dimensions

            #calculate new coordinate points
            left = nx / 2 - size / 2
            top = ny / 2 - size / 2
            right = nx / 2 + size / 2
            bottom = ny / 2 + size / 2

            # Crop the center of the image
            im_crop = im.crop((left, top, right, bottom))
            im_crop.save(f + ' centercrop.png', 'png', quality=90)

#crop_resize()

