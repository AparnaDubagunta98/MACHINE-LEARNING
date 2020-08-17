#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'


#SETTING UP THE TEST, TRAINA AND DEVELOPMENT DATASETS
features =[ 'Clump_thickness','Uniformity_of_cell_size', 'Uniformity_of_cell_shape','Marginal_adhesion', 
           'Epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin','normal_nucleoli', 'Mitoses']
columns = ["Clump_thickness", "Uniformity_of_cell_size", "Uniformity_of_cell_shape", 
           "Marginal_adhesion", "Epithelial_cell_size", "Bare_nuclei", "Bland_chromatin", 
           "normal_nucleoli", "Mitoses", "Class"]

data_train = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL 2019/CSCE 421/HW/HW1/hw1_question1_train.CSV", 
                   names = columns)

data_test = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL 2019/CSCE 421/HW/HW1/hw1_question2_test.CSV", 
                   names = columns)


data_dev = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL 2019/CSCE 421/HW/HW1/hw1_question2_dev.CSV", 
                   names = columns)


########################################## FUNCTION DEFINITIONS #######################################################


#IMPLEMENTATION OF KNN ALGORITHM

def knn(k,traindata,point,dist_type):   
    train = traindata[features].to_numpy() #create feature vector
    distances = dict()
    for i in range(len(train)):
        if dist_type == 'EU':    #EUCLIDEAN DISTANCE
            eu_dist = distance.euclidean(train[i],point).round(3) #calculate euclidean distance
            distances.update([(eu_dist,i)]) #append distance to dictionary with specific row number in train data
        elif dist_type == 'L1':  #L1-NORM
            l1_dist = np.linalg.norm((train[i] - point), ord=1)
            distances.update([(l1_dist,i)])
        elif dist_type == 'COS': #COSINE SIMILARITY
            cos_dist = distance.cosine(train[i],point).round(3)
            distances.update([(cos_dist,i)])

    nn = sorted(distances.items())  #Sort to find nearest neighbors
    nn = nn[:k] #find required k nearest neighbors
    class2 = 0  #votes for class 2 - benign
    class4 = 0  #votes for class 4 - malignant
    for i in range(k):
        dec = (traindata.iloc[nn[i][1]])['Class']  #find classification by traversing tarinign data
        if dec == 2:
            class2 +=1 #vote for class 2
        elif dec==4:
            class4 +=1 #vote for class 4
    #return classification decision / predicted label
    if class2 > class4:
        return 2
    else:
        return 4


#FUNCTION TO CALCULATE IMBALANCED ACCURACY
def acc(y_labels,y_dev):
    #formula used to calculate IMBALANCED ACCURACY
    correct = 0
    for i in range(len(y_dev)):
        if y_dev[i] == y_labels[i]:
            correct +=1
    
    acc = correct/len(y_dev)
    return acc

 
#FUNCTION TO CALCULATE BALANCED ACCURACY
def bacc(y_labels,y_dev):
    #formula used to calculate BALANCED ACCURACY
    class2 = 0     #total in class2
    class4 = 0     #total in class4
    correct_class2 = 0 #correctly in class4
    correct_class4 = 0 #correctly in class4
    for i in range(len(y_dev)):
        if y_labels[i] == 2:
            class2 +=1
            if y_dev[i] ==2:
                correct_class2 +=1
        elif y_labels[i] == 4:
            class4 +=1
            if y_dev[i] ==4:
                correct_class4 +=1
    
    bacc = 0.5*((correct_class2/class2)+(correct_class4/class4))
    return bacc
        



#FUNCTION TO PLOT METRICS ON SAME GRAPH    
def plot_metrics(accl,baccl):
    kvals = [x for x in range(1,20,2)]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('K-value')
    ax1.set_ylabel('accuracy metric', color=color)
    ax1.scatter(kvals, accl, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('B accuracy metric', color=color)  # we already handled the x-label with ax1
    ax2.scatter(kvals, baccl, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    


########################################## MAIN PROGRAM #######################################################

#2)ai- DATA EXPLORATION 

no_of_benign = len(data_train[data_train['Class'] == 2])
no_of_malignant = len(data_train[data_train['Class'] == 4])

print("Number of benign = ",no_of_benign)
print("Number of malignant = ",no_of_malignant)

#benign and malignant not equally distributed in the dataset ; approximately 3:1 ratio



#2)aii- DATA EXPLORATION 
features_set = data_train[['Clump_thickness','Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
                 'Marginal_adhesion', 'Epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin','normal_nucleoli', 'Mitoses']]

histograms = features_set.hist(figsize=(10,11),bins=5)
#not equally distributed


#2)aiii- DATA EXPLORATION 

pairs = [['Clump_thickness','Uniformity_of_cell_size'],['Bland_chromatin','Marginal_adhesion'],
         ['Uniformity_of_cell_shape','Epithelial_cell_size'],['Bare_nuclei','normal_nucleoli'],['Bland_chromatin','Mitoses']]

colors = {2:'red', 4:'blue'}

#2)b
for i in range(len(pairs)):
    df= data_train[pairs[i] + ['Class']]
    df.plot.scatter(x= pairs[i][0], y=pairs[i][1], c= df['Class'].map(colors))


dev = data_dev[features].to_numpy()

accuracy = dict()
baccuracy = dict()
y_labels = list(data_dev['Class'].to_numpy())

ch = 'EU'  #choice of distance type ; Use 'L1' for l1 norm ; Use 'COS' for Cosine similarity
 
#run for each k value from 1 to 19 (odd)
for k in range(1,20,2):
    cn = 'k='+str(k)
    data_dev[cn] = ""
    for i in range(len(data_dev)):
        label = knn(k,data_train,dev[i],ch) #get prediction
        data_dev[cn][i] = label 
    y_dev = list(data_dev[cn].to_numpy()) #get dev labels
    
    #compare with original labels and get accuracy metrics
    ac = acc(y_labels,y_dev)
    bac = bacc(y_labels,y_dev)
    accuracy.update([(k,ac)])
    baccuracy.update([(k,bac)])
    
#find best k value corresponding to highest value of accuracy metric
aclist = sorted(accuracy.items(), key = lambda x : x[1])
K1 = aclist[-1][0]

baclist = sorted(baccuracy.items(), key = lambda x : x[1])
K2 = baclist[-1][0]

plot_metrics(list(accuracy.values()),list(baccuracy.values()))

print("IMPLEMENTING KNN")
print("FOR DEVELOPMENT DATASET\n")
print("ACC LIST: ",list(accuracy.values()),"\n")
print("Balanced ACC LIST",list(baccuracy.values()),"\n\n")
        
    
hyperparameters = [K1,K2]
        
#TEST MODEL ON TEST DATA

#USE EACH K1 AND K2 TO TEST

print("TESTING THE MODEL\n")

test = data_test[features].to_numpy()
y_labels = list(data_test['Class'].to_numpy())
for j in range(len(hyperparameters)):
    test_labels =[]
    for i in range(len(data_test)):
        label = knn(hyperparameters[j],data_train,test[i],ch)  #get prediction
        test_labels.append(label) #get test labels
    
    #compare with original labels and get accuracy metrics
    ac2 = acc(y_labels,test_labels) 
    bac2 = bacc(y_labels,test_labels)
    print("For K",j+1," = ",hyperparameters[j])
    print("ACC: ",ac2,"BACC = ",bac2)

