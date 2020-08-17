#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import math

#SETTING UP THE TEST AND TRAIN DATASETS

columns = ['Frequency', 'AngleOfAttack','ChordLength','FreeStreamVelocity','SuctionDisplacement','ScaledSoundPressure']
features = ['Frequency', 'AngleOfAttack','ChordLength','FreeStreamVelocity','SuctionDisplacement']
data_train = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL_2019/CSCE 421/HW/HW2/airfoil_self_noise_train.csv", 
                   names = columns)

data_test = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL_2019/CSCE 421/HW/HW2/airfoil_self_noise_test.csv", 
                   names = columns)

dt = data_train[features]
dt.insert(0,'bias',1)

########### FUNCTION DEFINITIONS #############################

def linear_ols_model(dt,y):
   xlist = [list(dt.iloc[i]) for i in range(len(dt))]
   X = np.matrix(xlist)
   XT = X.transpose()
   
   xtx = np.dot(XT,X)
   xtx_inverse = np.linalg.pinv(xtx) #pseudo inverse
   xtxixt = np.dot(xtx_inverse,XT)
   
   w = np.dot(xtxixt,y)
   return w

def model_test(test,weights,feats):
    dt = test[feats]
    dt.insert(0,'bias',1)
    xlist = [list(dt.iloc[i]) for i in range(len(dt))]
    X = np.matrix(xlist)
    
    outcomes = test[['ScaledSoundPressure']]
    ylist = [list(outcomes.iloc[i]) for i in range(len(outcomes))]
    y = np.matrix(ylist)
    
    predictions = np.dot(X,weights)
    ybar = y - predictions
    ybart = ybar.transpose()
    
    RSS = np.dot(ybart,ybar)
    
    return (np.asscalar(RSS))
    
   
############# MAIN PROGRAM ##################
    
#2)i
histograms = data_train.hist(figsize=(10,11),bins=5)

#2)ii
outcomes = data_train[['ScaledSoundPressure']]
ylist = [list(outcomes.iloc[i]) for i in range(len(outcomes))]
y = np.matrix(ylist)

# training model
weights = linear_ols_model(dt,y)


# testing model and finding RSS errors using the model on test data
#model_test(data_test,weights,features)
RSS = model_test(data_test,weights,features) #output will be square root of this value
print("For All 5 features :")
print("WEIGHTS MODEL: ",weights.tolist())
print("Square root of RSS ERROR VALUE FOR MODEL ON TEST DATA IS : ",round(math.sqrt(RSS),3),"\n")

feature_sets = [['Frequency', 'SuctionDisplacement','ChordLength'],['FreeStreamVelocity','SuctionDisplacement'],
                ['Frequency','FreeStreamVelocity'],['AngleOfAttack','ChordLength']]

for f in feature_sets:
    dt = data_train[f]
    dt.insert(0,'bias',1)
    w = linear_ols_model(dt,y)
    RSS = model_test(data_test,w,f)
    
    print("For features :",','.join(f))
    print("WEIGHTS MODEL: ",w.tolist())
    print("Square root of RSS ERROR VALUE FOR MODEL ON TEST DATA IS : ",round(math.sqrt(RSS),3),"\n")
    
    
