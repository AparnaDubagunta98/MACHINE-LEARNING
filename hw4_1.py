#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import random
import math
from sklearn.tree import DecisionTreeClassifier 



data_train = pd.read_csv("OnlineNewsPopularityTrain.csv", usecols = [i for i in range(1,61)])
data_test = pd.read_csv("OnlineNewsPopularityTest.csv", usecols = [i for i in range(1,61)])
features = data_train.columns[:59]

indices = [i for i in range(0,38422)]



def crossval(dft,folds):
    rs = dft.shape[0]
    div = math.ceil(rs/folds)
    no = math.ceil(rs/folds)
    splits = []
    random.shuffle(indices)
    for i in range(0,len(indices),no):
        l = indices[i:div]
        splits.append(l)
        div += no
    return splits

def compute_accuracies(ytest,ypred):
    rsum = 0
    for i in range(len(ytest)):
        diff = ytest[i] - ypred[i]
        rsum += diff**2
    return math.sqrt(rsum)


    
splits = crossval(data_train,5)  
depths = []
for depth in range(2,6):
        s_accuracies = []
        for s in splits:
            trainin = s
            testin = [i for i in indices if i not in s]
            X_Train_s = pd.DataFrame(columns = features)
            Y_Train_s = pd.DataFrame(columns = [' shares'])
            X_Test = pd.DataFrame(columns = features)
            Y_Test = pd.DataFrame(columns = [' shares'])
            
            for i in trainin:
                X_Train_s.loc[i] = data_train.iloc[i]
                Y_Train_s.loc[i] = data_train.iloc[i]
        
            for j in testin:
                X_Test.loc[j] = data_train.iloc[j]
                Y_Test.loc[j] = data_train.iloc[j]
            
            tree = DecisionTreeClassifier(criterion='entropy',max_depth=depth, random_state=1)
            tree.fit(X_Train_s, Y_Train_s)
            
            Y_pred = list(tree.predict(X_Test))
            s_acc = compute_accuracies(list(Y_Test[' shares']),Y_pred)
            s_accuracies.append(round(s_acc,2))
         
        depth_average_accuracy = round((sum(s_accuracies)/len(s_accuracies)),3)
        depths.append((depth,depth_average_accuracy))

dict_depths = dict()
for i in depths:
    dict_depths[i[1]] = i[0]


optimal_depth = dict_depths[min(dict_depths.keys())]
tree = DecisionTreeClassifier(criterion='entropy',max_depth=optimal_depth, random_state=1)
X_Test = data_test[features]
Y_Test = data_test[[' shares']]
tree.fit(X_Test,Y_Test)
Y_pred = list(tree.predict(X_Test))
final_acc = round(compute_accuracies(list(Y_Test[' shares']),Y_pred),2)

importances = tree.feature_importances_
print(importances)