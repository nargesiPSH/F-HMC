import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import chainer
import random
from PIL import Image
from missingpy import KNNImputer
from missingpy import MissForest
from fancyimpute import IterativeImputer
from ppca import PPCA
from sklearn.metrics import mean_squared_error
# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist(ndim=1)


def showImagesRandomImages(n):
# this function show n number of images from MNIST data set and returns the last one
    for i in range(n):
        index = random.randint(0, 100)
        image, label = test[index]
        image1 = image.reshape(28, 28)
        plt.imshow(image1, cmap='gray', vmin=0, vmax=1)
        plt.show()
        return image1

def generateMissingFig(Org,P):
# this function generates Images with missing values given Org = orinigal image and P = missing percentage
    Mask = np.zeros((28, 28))
    out = deepcopy(Org)
    rgbArray = np.zeros((28, 28, 3), 'uint8')
    rgbArray[..., 0] = Org * 256
    rgbArray[..., 1] = Org * 256
    rgbArray[..., 2] = Org * 256
    original = Image.fromarray(rgbArray)
    plt.imshow(original)
    Missingnumbers = int(P * 784)
    for i in range(Missingnumbers):
        index1 = random.randint(0,27)
        index2 = random.randint(0,27)
        rgbArray[index1,index2] = (0,0,255)
        out[index1,index2] = np.nan
        Mask[index1,index2] = 0
    Missing = Image.fromarray(rgbArray)
    plt.imshow(Missing)
    plt.show()
    return out

SelectedImage = showImagesRandomImages(3)  #select and image randomly from MNSIT dataset
missingPercentage = 0.2  # missing rate percentage
missingImage = generateMissingFig(SelectedImage,missingPercentage) #inserting missing values to the original image

imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputed_by_KNN = imputer.fit_transform(missingImage)
KNNImputed_RMSE = mean_squared_error(SelectedImage, imputed_by_KNN)
#plt.imshow(imputed_by_KNN, cmap='gray', vmin=0, vmax=1)
#plt.show()

imputer = MissForest()
MissForest_imputed = imputer.fit_transform(missingImage)
MissForest_RMSE = mean_squared_error(SelectedImage, MissForest_imputed)
#plt.imshow(MissForest_imputed, cmap='gray', vmin=0, vmax=1)
#plt.show()


imputer = IterativeImputer()
MICE_imputed = imputer.fit_transform(missingImage)
MICE_RMSE = mean_squared_error(SelectedImage, MICE_imputed)
#plt.imshow(MICE_imputed, cmap='gray', vmin=0, vmax=1)
#plt.show()

ppca = PPCA()
ppca.fit(data=SelectedImage, d=100, verbose=True)
PPCA_imputed= ppca.transform(missingImage)
PPCA_RMSE = mean_squared_error(SelectedImage, PPCA_imputed)
#plt.imshow(PPCA_imputed, cmap='gray', vmin=0, vmax=1)
#plt.show()