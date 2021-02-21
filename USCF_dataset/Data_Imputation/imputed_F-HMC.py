import xlrd
from autograd import grad
import autograd.numpy as np
import scipy.stats as st
import random
from copy import deepcopy
import xlrd
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import KernelDensity
import pandas as pd
import dateutil.parser as dparser
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import datetime
from scipy.stats import norm
from math import sqrt
import xlsxwriter
import seaborn as sns
from sklearn.mixture import GaussianMixture
import pymc3 as pm
from scipy.stats import bernoulli
from scipy.stats import multinomial
import theano.tensor as tt
from theano.tensor import _shared
import theano
from cycler import cycler

# ............................................ Reading Data from data set ..............................................
# read original data with missing values and generated data using F-HMC

# reading generated data from F-HMC
generated = pd.read_excel('generated2.xlsx')
header = generated.columns.values

original = pd.read_excel('extracFeatures2.xlsx', usecols=header)

#Spotting missing vlaues
original = original.replace(-1,np.nan)
row, col = original.shape
#..............................................Generatitng Mask matrix to identify missign values.......................
DataMissingMask = np.zeros((row, col))
for i in range (row):
    for j in range(col):
        test = original.iloc[i,j]
        if(pd.isna(test)):
            DataMissingMask[i][j] = 1


# ...........................................Marginalization for missing values over generated data ..................

Imputed = original
for p in range(row):
    idxgenerated = np.random.randint(len(generated), size=500)
    selectedGenerated = generated.iloc[idxgenerated, :]
    for k in range(col):
        if (DataMissingMask[p][k] == 1):
            marginalization = selectedGenerated.mean(axis = 0).to_numpy()
            Imputed.iloc[p,k] = marginalization[k]
#..................................................  Write finanl imputed data ...............................
Imputed.to_excel("imputedGen.xlsx")

print("imputating dataset Finished!")