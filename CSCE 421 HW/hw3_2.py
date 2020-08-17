#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import math
import sympy as sym

data_train = pd.read_csv("/Users/aparnadubagunta/Desktop/FALL_2019/CSCE 421/HW/HW3/hw3_question2.csv")

hours = [i for i in range(0,data_train.shape[0])]
tickets = list(data_train['Count'])

#2.i
plt.plot(hours,tickets)
plt.xlabel('hours elapsed')
plt.ylabel('Number of tickets sold')
plt.show()

#2.ii
#ii.a
month1 = data_train[0:744]
m1hrs = [i for i in range(744)]
month1_tickets = list(month1['Count'])

plt.plot(m1hrs,month1_tickets)
plt.xlabel('hours elapsed')
plt.ylabel('Number of tickets sold')
plt.show()

#ii.b

likelihoods = []
x = sym.Symbol('x')
lda = sym.Symbol('lda')
f_x  =((sym.exp(-lda))* pow(lda,x))/(sym.factorial(x))


for i in range(744):
    f= f_x.subs(x,month1_tickets[i])
    likelihoods.append(f)

l_lda = np.prod(likelihoods)
print(l_lda)

#ii.c
negLogL_lda = -(sym.log(l_lda))
print(negLogL_lda)

#ii.d
dnlog_lda = sym.diff(negLogL_lda,lda)
print(dnlog_lda)


solution_lda = round((sym.solve(dnlog_lda,lda))[0],2)

#ii.e
print(solution_lda)




    
    

