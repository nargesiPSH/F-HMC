#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # NOTE: This was tested with matplotlib v. 2.1.0
import xlrd
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
import seaborn as sns
sns.set()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

#importing our cancer dataset

#....................................................... Reading Data ..................................................
Features1 = xlrd.open_workbook("outputOrg.xlsx") #  'originalFulloccurance.xlsx'
sheet1 = Features1.sheet_by_index(0)
datasetO = np.zeros((sheet1.nrows-1,sheet1.ncols))
FeatureIndexesTitle = []
for i in range(1,sheet1.nrows):
    for j in range(sheet1.ncols):
        temp = sheet1.cell_value(i, j)
        datasetO[i-1][j] = int(temp) #ignoring feature title by starting from 1 and but storage in array starts from 0 which means i-1

x = datasetO[:, 0:37]
restX = datasetO[:,39:]
XO = np.concatenate((x, restX), axis=1)
YO = datasetO[:, 38]


Features2 = xlrd.open_workbook("outputGen.xlsx") 
sheet2 = Features2.sheet_by_index(0)
datasetS = np.zeros((sheet2.nrows-1,sheet1.ncols))
FeatureIndexesTitle = []

for i in range(1,sheet2.nrows):
    for j in range(sheet2.ncols):
        temp = sheet2.cell_value(i, j)
        datasetS[i-1][j] = int(temp)
FeatureIndexesTitle = np.asarray(FeatureIndexesTitle)
x = datasetS[:, 0:37]
restX = datasetS[:,39:]
XS = np.concatenate((x, restX), axis=1)
YS = datasetS[:, 38]

# Splitting the dataset into the Training set and Test set
def classifiersAcurracy(A,B):
    #Feature Scaling
    sc = StandardScaler()
    X_train = A #sc.fit_transform(X)
    Y_train = B
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    Models = []
    kfoldsNumber = 10
    cv = StratifiedKFold(n_splits=kfoldsNumber)



    #Using Logistic Regression Algorithm to the Training Set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
       

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro' , labels = [1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("Logistic Regression")

    d = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df = pd.DataFrame(data=d)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []

    #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
      

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("Knn")

    d2 = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df2 = pd.DataFrame(data=d2)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    df = df.append(df2, ignore_index=True)

    #Using SVC method of svm class to use Support Vector Machine Algorithm
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
    

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("Linear SVM")

    d3 = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df3 = pd.DataFrame(data=d3)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    df = df.append(df3, ignore_index=True)


    #Using SVC method of svm class to use Kernel SVM Algorithm
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
      

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("RBF SVM")

    d4 = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df4 = pd.DataFrame(data=d4)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    df = df.append(df4, ignore_index=True)




    #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
    

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("Decision Tree")

    d5 = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df5 = pd.DataFrame(data=d5)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    df = df.append(df5, ignore_index=True)


    #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        classifier.fit(X_train[train], Y_train[train])
        Y_pred = classifier.predict(X_train[test])
     

        precision = metrics.precision_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])
        accuracy = metrics.accuracy_score(Y_train[test], Y_pred)
        recall = metrics.recall_score(Y_train[test], Y_pred, average='macro')
        f1Score = metrics.f1_score(Y_train[test], Y_pred, average='macro', labels=[1, 2, 3, 4])

        acurracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(f1Score)
        Models.append("Random Forest")

    d6 = {'Accuracy': acurracies, 'Precision': precisions , 'Recall':recalls, 'F1 Score' : f1Scores, 'Classification Model':Models }
    df6 = pd.DataFrame(data=d6)
    Models = []
    acurracies = []
    precisions = []
    recalls = []
    f1Scores = []
    df = df.append(df6, ignore_index=True)

    yerrorss = []
    accuracyOriginalLfold = []

    yerrorss1 = []
    recallOriginalLfold = []

    yerrorss2 = []
    precisionOriginalLfold = []

    yerrorss3 = []
    f1scoreOriginalLfold = []

    acurracies = np.asarray(acurracies)
    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)
    f1Scores = np.asarray(f1Scores)

    for i in range(int(len(acurracies)/kfoldsNumber)):
        select = acurracies[i*kfoldsNumber:(i*kfoldsNumber)+kfoldsNumber]
        accuracyOriginalLfold.append(np.mean(select))
        yerrorss.append(np.std(select))

        select = recalls[i * kfoldsNumber:(i * kfoldsNumber) + kfoldsNumber]
        recallOriginalLfold.append(np.mean(select))
        yerrorss1.append(np.std(select))

        select = precisions[i * kfoldsNumber:(i * kfoldsNumber) + kfoldsNumber]
        precisionOriginalLfold.append(np.mean(select))
        yerrorss2.append(np.std(select))

        select = f1Scores[i * kfoldsNumber:(i * kfoldsNumber) + kfoldsNumber]
        f1scoreOriginalLfold.append(np.mean(select))
        yerrorss3.append(np.std(select))

    return accuracyOriginalLfold,yerrorss,recallOriginalLfold,yerrorss1,precisionOriginalLfold,yerrorss2,f1scoreOriginalLfold,yerrorss3, df

acurracyOnOriginal , errorOnoriginal, recallOnOriginal , recallerrorOnoriginal, precisionOnOriginal , precisionerrorOnoriginal , f1scoreOnOriginal , f1scoreerrorOnoriginal, DFO  = classifiersAcurracy(XO,YO)
acurracyOnSynthetic , errorOnSynthetic, recallOnSynthetic , recallerrorOnSynthetic, precisionOnSynthetic , precisionerrorOnSynthetic, f1scoreOnSynthetic , f1scoreerrorOnSynthetic , DFS = classifiersAcurracy(XS,YS)
xpltVlaues = ['Logistic Regression','KNN', 'Linear SVM', 'RBF SVM','Gaussian Bayes' , 'Decision Tree', 'Random Forest'] 



width = 0.27
ind = np.arange(len(acurracyOnOriginal))

dataFrameAll  = DFO.append(DFS, ignore_index = True)
L = []
for i in range(len(dataFrameAll.index)):
    if i < (len(dataFrameAll.index)/2):
        L.append("Original")
    else:
        L.append("Synthetic")

dataFrameAll["Type"] = L


# Creating plot
a = sns.boxplot(y='Accuracy', x='Classification Model',
                 data=dataFrameAll,
                 palette="colorblind",
                 hue='Type')
a.set_xlabel("Classification Model",fontsize=30)
a.set_ylabel("Accuracy Score",fontsize=30)
a.tick_params(labelsize=20)
a.legend(fontsize=20)
plt.show()

b = sns.boxplot(y='Precision', x='Classification Model',
            data=dataFrameAll,
            palette="colorblind",
            hue='Type')
b.set_xlabel("Classification Model",fontsize=30)
b.set_ylabel("Precision Score",fontsize=30)
b.tick_params(labelsize=20)
b.legend(fontsize=20)
plt.show()

c = sns.boxplot(y='Recall', x='Classification Model',
            data=dataFrameAll,
            palette="colorblind",
            hue='Type')
c.set_xlabel("Classification Model",fontsize=30)
c.set_ylabel("Recall Score",fontsize=30)
c.tick_params(labelsize=20)
c.legend(fontsize=20)
plt.show()

d = sns.boxplot(y='F1 Score', x='Classification Model',
            data=dataFrameAll,
            palette="colorblind",
            hue='Type')
d.set_xlabel("Classification Model",fontsize=30)
d.set_ylabel("F1 Score Score",fontsize=30)
d.tick_params(labelsize=20)
d.legend(fontsize=20)
plt.show()

