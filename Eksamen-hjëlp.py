# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:03:46 2018

@author: henni
"""
import numpy as np
import math
#Remember to import
#Mark the code and ctrl + enter (Selma: cmd + enter (perhaps))
def varPCA(n):
    #The array called S typically
    S = np.array([149,118,53,42,3])
    Var = 0
    for i in range(0,np.size(S)):
        Var = Var + S[i]**2
    for i in range(0,np.size(S)):
        VarX = S[i]**2/Var
        print("component {:d} accounts for: {:.2f}".format(i,VarX))
        
    return Var


#Mark the code and ctrl + enter (Selma: cmd + enter (perhaps))
def adaboost(n):
    #Number of observations
    n = 6
    errorRate = 2/6
    missclass = 2
    correctclass = 4
    
    #The formelu used
    alpha1 = 0.5 * math.log((1-errorRate)/errorRate)
    wm = 1/n * math.exp(-alpha1)
    wmMiss = 1/n * math.exp(alpha1)
    
    #Updated weight for missclassified observations
    wWrong = wmMiss/(missclass*wmMiss + correctclass*wm)
    #Updated weight for correct classified observations
    wCorrect = wm/(missclass*wmMiss + correctclass*wm)
    print("New weight for correct classified: {:.3f}".format(wCorrect))
    print("New weight for uncorrect classified: {:.3f}".format(wWrong))
    
    return wCorrect
    