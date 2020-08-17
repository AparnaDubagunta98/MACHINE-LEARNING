#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import random
import math
from sklearn.ensemble import RandomForestClassifier


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
depth = [2,3,5]
numtrees = [2,3,4,5,6]
depths = [[0 for x in range(len(numtrees))] for y in range(len(depth))] 
for d in range(len(depth)):
    for n in range(len(numtrees)):
        #5 fold
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
            
            tree = RandomForestClassifier(max_depth = depth[d],n_estimators=numtrees[n], bootstrap = True,max_features = 'sqrt')
            tree.fit(X_Train_s, Y_Train_s)
            
            Y_pred = list(tree.predict(X_Test))
            s_acc = compute_accuracies(list(Y_Test[' shares']),Y_pred)
            s_accuracies.append(round(s_acc,2))
         
        depth_average_accuracy = round((sum(s_accuracies)/len(s_accuracies)),3)
        depths[d][n] = depth_average_accuracy

H = np.array(depths)
fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

###Testing###
dss = [(min(depths[j]),depths[j].index(min(depths[j])),j) for j in range(len(depths))]

optimal_depth = depth[min(dss)[1]]
optimal_num_trees = numtrees[min(dss)[2]]

X_Test = data_test[features]
Y_Test = data_test[[' shares']]

tree = RandomForestClassifier(max_depth = optimal_depth,n_estimators=optimal_num_trees, bootstrap = True,max_features = 'sqrt')
tree.fit(X_Test, Y_Test)

Y_pred = list(tree.predict(X_Test))
final_acc = round(compute_accuracies(list(Y_Test[' shares']),Y_pred),2)






