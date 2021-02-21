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
# ................................ Reading Data from data set ..........................................................
global targetCancerType
global targetGender
global numberOfSmapleTarget
targetCancerType = 5
targetGender = 3
numberOfSmapleTarget = 500

# ................................ Write the finial outcome in file ....................................................
def WriteResultOnFile(trace):
#write the outcome on the file for furthure process
    traceouts = trace.get_values('out', burn=5000-numberOfSmapleTarget, combine=False)  # for getting 1330 sample
    traceout1 = np.asarray(traceouts[0])
    traceout2 = np.asarray(traceouts[1])

    workbook = xlsxwriter.Workbook('generated1CancertType'+str(targetCancerType)+ 'Gender'+str(targetGender)+'.xlsx')
    worksheet = workbook.add_worksheet()
    rowsNumber, FeatureNumbers = traceout1.shape
    for i in range(rowsNumber):
        for j in range(FeatureNumbers):
            if (i == 0):
                worksheet.write(i, j, FeatureoccuranceTitle[j])
            else:
                valuess = round(traceout1[i][j])
                if (j < 38):
                    if (valuess <= 0):
                        valuess = 0
                    if (valuess >= 1):
                        valuess = 1
                elif (j == 38):
                    if (valuess <= 1):
                        valuess = 1
                    if (valuess >= 4):
                        valuess = 4
                elif (j == 39):
                    if (valuess <= 0):
                        valuess = 1
                    elif (valuess >= 1):
                        valuess = 2
                worksheet.write(i, j, valuess)
    workbook.close()

    workbook = xlsxwriter.Workbook('generated2CancertType'+str(targetCancerType)+ 'Gender'+str(targetGender)+'.xlsx')
    worksheet = workbook.add_worksheet()
    rowsNumber, FeatureNumbers = traceout2.shape
    for i in range(rowsNumber):
        for j in range(FeatureNumbers):
            if (i == 0):
                worksheet.write(i, j, FeatureoccuranceTitle[j])
            else:
                valuess = round(traceout2[i][j])
                if (j < 38):
                    if (valuess <= 0):
                        valuess = 0
                    if (valuess >= 1):
                        valuess = 1
                elif (j == 38):
                    if (valuess <= 1):
                        valuess = 1
                    if (valuess >= 4):
                        valuess = 4
                elif (j == 39):
                    if (valuess <= 0):
                        valuess = 1
                    elif (valuess >= 1):
                        valuess = 2
                worksheet.write(i, j, valuess)
    workbook.close()
# ................................ setup function to read the input files and preprocessing ............................
def mysetup():
    global targetCancerType
    global targetGender
    global numberOfSmapleTarget
    Features = xlrd.open_workbook('extracFeatures2.xlsx')
    sheet = Features.sheet_by_index(0)
    DataAccess = np.zeros((sheet.nrows,sheet.ncols))

    FeatureIndexesTitle = []

    for i in range(sheet.nrows):
        for j in range(sheet.ncols):
            temp = sheet.cell_value(i, j)
            if(i == 0):
                FeatureIndexesTitle.append(temp)
            elif(temp == ' '):
                DataAccess[i][j] = -99
            else:
                DataAccess[i][j] = int(temp)

    symptomIndexes = []

    for i in range(39):
        if(i< 38):
            symptomIndexes.append(i*4)
        else:
            symptomIndexes.append(152) #cancerType
            symptomIndexes.append(153) #gender
            symptomIndexes.append(154) #age
    SymptomIndexesTarget = []

    if(targetGender == 3 and targetCancerType == 5):
        SymptomIndexesTarget =  np.where(DataAccess[:,152] != 12)[0]#
    elif(targetGender == 3 and targetCancerType != 5):
        SymptomIndexesTarget = np.where(DataAccess[:,152] == targetCancerType)[0]
    elif(targetGender != 3 and targetCancerType == 5):
        SymptomIndexesTarget = np.where(DataAccess[:, 153] == targetGender)[0]
    elif(targetGender != 3 and targetCancerType != 5):
        print("Invalid configuratuin the selected groups of gender and cancer type is too small!")


    dataTarget = DataAccess[SymptomIndexesTarget,:]

    FeatureIndexesTitle = np.asarray(FeatureIndexesTitle)
    FeatureoccuranceTitle = FeatureIndexesTitle[symptomIndexes]

    DataAccessSymptopmsOnly = dataTarget[:,symptomIndexes]
    DataAccessSymptopmsOnly = np.where(DataAccessSymptopmsOnly == -1, np.nan, DataAccessSymptopmsOnly)
    DataAccessProcessedSymptopmsOnly = DataAccessSymptopmsOnly[~np.isnan(DataAccessSymptopmsOnly).any(axis=1)]

    numberOfSmapleTarget = len(DataAccessProcessedSymptopmsOnly)

    rowsNumber1,FeatureNumbers1 = DataAccessProcessedSymptopmsOnly.shape
    workbook = xlsxwriter.Workbook('originalFulloccuranceCancertType'+str(targetCancerType)+ 'Gender'+str(targetGender)+'.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    for i in range(rowsNumber1):
        for j in range(FeatureNumbers1):
            if(i == 0):
                worksheet.write(i, j, FeatureoccuranceTitle[j])
            else:
                worksheet.write(i, j, DataAccessProcessedSymptopmsOnly[i][j])
    workbook.close()


    meanValues = np.mean(DataAccessProcessedSymptopmsOnly, axis=0)
    maxAge = np.max(DataAccessProcessedSymptopmsOnly[:,-1])

    GenderCopy = DataAccessSymptopmsOnly[:, -2]
    indexes = np.where(GenderCopy == 1)
    gender1success = np.sum(GenderCopy[indexes]) / (len(GenderCopy))

    cancerTypeCopy = DataAccessSymptopmsOnly[:,-3]

    indexes = np.where(cancerTypeCopy == 1)
    type1success = np.sum(cancerTypeCopy[indexes])/(len(cancerTypeCopy))
    indexes = np.where(cancerTypeCopy == 2)
    type2success = np.sum(cancerTypeCopy[indexes]) /(2*len(cancerTypeCopy))
    indexes = np.where(cancerTypeCopy == 3)
    type3success = np.sum(cancerTypeCopy[indexes]) /(3*len(cancerTypeCopy))
    indexes = np.where(cancerTypeCopy == 4)
    type4success = np.sum(cancerTypeCopy[indexes]) /(4*len(cancerTypeCopy))

    return symptomIndexes,meanValues,gender1success,type1success,type2success,type3success,type4success,maxAge,meanValues,DataAccessProcessedSymptopmsOnly,FeatureoccuranceTitle

# ................................ The core body of the algorithm ......................................................
if __name__ == '__main__':
    symptomIndexes, meanValues, gender1success, type1success, type2success, type3success, type4success, maxAge, meanValues, DataAccessProcessedSymptopmsOnly,FeatureoccuranceTitle = mysetup()
    syntheticdata_model = pm.Model()

    with syntheticdata_model:
        symps = np.zeros((1, 38))[0]
        for i in range(len(symptomIndexes) - 3):  # except last three features gender and age
            symps[i] = pm.Bernoulli('symp' + str(i), p=meanValues[i]).random()  # values not just probabilities

        sympsTheano = theano.shared(np.array(symps).astype("float64"))
        genderType = pm.Bernoulli('gender', p=meanValues[i]).random()

        true_probs = [type1success, type2success, type3success, type4success]
        true_probs1 = pm.Normal('p1' , mu=type1success, sigma=0.01)
        true_probs2 = pm.Normal('p2', mu=type2success, sigma=0.01)
        true_probs3 = pm.Normal('p3', mu=type3success, sigma=0.01)
        true_probs4 = pm.Normal('p4', mu=type4success, sigma=0.01)
        cancerTypeValues = [1, 2, 3, 4]
        cancerType = pm.Multinomial('cancer type', n=1, p=[type1success, type2success ,type3success, type4success], shape=4).random()
        indexCancer = np.where(cancerType == 1)[0]
        cancerTypeGeneratedSample = cancerTypeValues[indexCancer[0]]
        age = pm.Normal('age', mu=meanValues[-1], sigma=(maxAge - meanValues[-1])).random()
        cov = np.cov(DataAccessProcessedSymptopmsOnly.T) #
        featurenumber = 41

        x = pm.math.stack(cancerTypeGeneratedSample, genderType, age)
        allMu = pm.math.concatenate([sympsTheano, x], axis=0)
        test = pm.MvNormal('out', mu=allMu, cov=cov, shape=featurenumber)
        returns = test
        step = pm.HamiltonianMC()
        trace = pm.sample(5000, step=step, chains=2) #,init='adapt_diag' # #, cores=1, chains=1

    plt.figure()
    # plot the trace file of the algorithm
    traceArray = trace['out']
    pm.traceplot(trace, var_names=['age', 'gender', 'symp1'] , compact=False)
    plt.show()
    traceArray = trace['out']

    # plot the covariance fo original and generated data
    covv = np.cov(traceArray.T)
    # plt.show()
    fig = plt.figure()
    ax = sns.heatmap(covv[1:-3, 1:-3], center=0)  #
    #plt.show()

    fig = plt.figure()
    ax = sns.heatmap(cov[1:-3, 1:-3], center=0)  #
    #plt.show()
    #WriteResultOnFile(trace=trace)

    # plot 3 symptoms of dataset for visualization purpose
    age = trace.get_values('age', combine=False)
    gender = trace.get_values('gender', combine=False)
    symp1 = trace.get_values('symp1', combine=False)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)

    ax0.hist(age, bins=100, label=['Original', 'Generated'], alpha=0.5, orientation="horizontal");
    ax1.hist(gender, bins=2, label=['Original', 'Generated'], alpha=0.5, orientation="horizontal");
    ax2.hist(symp1, bins=2, label=['Original', 'Generated'], alpha=0.5, orientation="horizontal");
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.show()
    print("Done")




