import numpy as np
import pandas as pd
import sklearn.naive_bayes as NB

data = pd.read_csv('/Users/snehamitta/Desktop/ML/Assignment4/Purchase_Likelihood.csv')

def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pd.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')

   #print("Frequency Table: \n")
   #print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return
   
gs_table = TargetPercentByNominal(data.group_size, data.A)
ho_table = TargetPercentByNominal(data.homeowner, data.A)
mc_table = TargetPercentByNominal(data.married_couple, data.A)

nTotal = len(data)

#Q2.a) The class probabilities 

crossTable = pd.crosstab(index = data['A'], columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = (crossTable['Count'] / nTotal)
crossTable = crossTable.drop(columns = ['All'])
print(crossTable)

#Q2.b) group_size = 1, homeowner = 0, and married_couple = 0
a0 = 0.803530*0.547418*0.815013*0.215996
a1 = 0.773475*0.429815*0.782206*0.640462
a2 = 0.778010*0.489407*0.788661*0.143542
tot = a0+a1+a2
a0 = a0/tot
print('pr(A = 0) is',a0)
a1 = a1/tot
print('pr(A = 1) is',a1)
a2 = a2/tot
print('pr(A = 2) is',a2)

#Q2.c) group_size = 2, homeowner = 1, and married_couple = 1
a0 = 0.179051*0.452582*0.184987*0.215996
a1 = 0.213734*0.570185*0.217794*0.640462
a2 = 0.205255*0.510593*0.211339*0.143542
tot = a0+a1+a2
a0 = a0/tot
print('pr(A = 0) is',a0)
a1 = a1/tot
print('pr(A = 1) is',a1)
a2 = a2/tot
print('pr(A = 2) is',a2)

#Q2.d) group_size = 3, homeowner = 1, and married_couple = 1
a0 = 0.015881*0.452582*0.184987*0.215996
a1 = 0.011897*0.570185*0.217794*0.640462
a2 = 0.015761*0.510593*0.211339*0.143542
tot = a0+a1+a2
a0 = a0/tot
print('pr(A = 0) is',a0)
a1 = a1/tot
print('pr(A = 1) is',a1)
a2 = a2/tot
print('pr(A = 2) is',a2)

#Q2.e) group_size = 4, homeowner = 0, and married_couple = 0
a0 = 0.001538*0.547418*0.815013*0.215996
a1 = 0.000894*0.429815*0.782206*0.640462
a2 = 0.000974*0.489407*0.788661*0.143542
tot = a0+a1+a2
a0 = a0/tot
print('pr(A = 0) is',a0)
a1 = a1/tot
print('pr(A = 1) is',a1)
a2 = a2/tot
print('pr(A = 2) is',a2)











