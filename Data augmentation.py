import cv2
import numpy as np
from skimage.feature import hog
import os
import glob
from scipy import ndimage, misc
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

samples = []
labels = []

path = r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\Training InAsBi - Copy"
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

deskew()

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

            #calculate new coordinate points
            left = nx / 2 - 10
            top = ny / 2 - 10
            right = nx / 2 + 10
            bottom = ny / 2 + 10

            # Crop the center of the image
            im_crop = im.crop((left, top, right, bottom))
            im_crop.save(f + ' centercrop.png', 'png', quality=90)

crop_resize()

# # Get positive samples
# pos_im_path = r"C:\Users\s169261\Documents\`BEP\Alessandro\QDs\Trainingdsk1"
# for filename in glob.glob(os.path.join(pos_im_path, '*.jpg')):
#     img = cv2.imread(filename, 1)
#     img = np.resize(img, (64, 64))
#     hist = hog(img)
#     samples.append(hist)
#     labels.append(1)
#
# # Get negative samples
# neg_im_path= r"C:\Users\s169261\Documents\`BEP\Alessandro\Clocks\Positive\Clocks"
# for filename in glob.glob(os.path.join(neg_im_path, '*.jpg')):
#     img = cv2.imread(filename, 1)
#     hist = hog(img)
#     samples.append(hist)
#     labels.append(0)
#
# # Convert objects to Numpy Objects
# samples = np.float32(samples)
# labels = np.array(labels)
#
#
# # Shuffle Samples
# rand = np.random.RandomState(321)
# shuffle = rand.permutation(len(samples))
# samples = samples[shuffle]
# labels = labels[shuffle]
#
# # Create SVM classifier
# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR
# # svm.setDegree(0.0)
# svm.setGamma(5.383)
# # svm.setCoef0(0.0)
# svm.setC(2.67)
# # svm.setNu(0.0)
# # svm.setP(0.0)
# # svm.setClassWeights(None)
#
# # Train
# svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
# svm.save('svm_data.dat')