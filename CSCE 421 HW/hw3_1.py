#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import math
import random
from sklearn.linear_model import LogisticRegression 


def compute_accuracies(ytest,ypred):
    correct = 0
    for i in range(len(ypred)):
        if ypred[i] == ytest[i]:
            correct +=1
    
    acc = correct/len(ypred)
    return acc
     



#SETTING UP THE TEST AND TRAIN DATASETS
columns = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
features = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
data_train = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL_2019/CSCE 421/HW/HW3/hw3_question1.csv")



#1. a: DATA EXPLORATION
histo = data_train.hist('area',bins=50)

#1. b

data_train['class'] =0
columns.append('class')

#### dichomotizing outcomes ####

for i in range(data_train.shape[0]):
    if data_train.iloc[i]['area'] == 0.0:
        data_train['class'].iloc[i]= 0
    elif data_train.iloc[i]['area'] >0.0:
        data_train['class'].iloc[i]= 1


##### creating splits of 10 ######

indices = [i for i in range(0,466)]
div = 47
splits = []
random.shuffle(indices)
for i in range(0,len(indices),47):
    l = indices[i:div]
    splits.append(l)
    div += 47

#########  Running Logostic Regression with Cross Vlaidation #######
s_accuracies = []
for s in splits:
    print("hi")
    trainin = s
    testin = [i for i in indices if i not in s]
    
    X_Train_s = pd.DataFrame(columns = features)
    Y_Train_s = pd.DataFrame(columns = ['class'])
    X_Test = pd.DataFrame(columns = features)
    Y_Test = pd.DataFrame(columns = ['class'])
    
    for i in trainin:
        X_Train_s.loc[i] = data_train.iloc[i]
        Y_Train_s.loc[i] = data_train.iloc[i]
    
    for j in testin:
        X_Test.loc[j] = data_train.iloc[j]
        Y_Test.loc[j] = data_train.iloc[j]
    
#    classifier = LogisticRegression(random_state = 0) 
#    classifier.fit(X_Train_s, Y_Train_s) 
#    
#    Y_pred = list(classifier.predict(X_Test))
#    
#    s_acc = compute_accuracies(list(Y_Test['class']),Y_pred)
#    s_accuracies.append(round(s_acc,2))

#average_accuracy = round((sum(s_accuracies)/len(s_accuracies)),3)
#print("Accuracies for each of the ten folds: ",s_accuracies)
#print("Average Accuracy for all 10 folds: ",average_accuracy)








    

