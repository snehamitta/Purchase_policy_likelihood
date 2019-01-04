import numpy
import pandas as pd

data = pd.read_csv('/Users/snehamitta/Desktop/ML/Assignment4/Purchase_Likelihood.csv')

def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pd.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')

   
   print("Percent Table: \n")
   print(percentTable)

   return percentTable
   
gs_table = TargetPercentByNominal(data.group_size, data.A)
ho_table = TargetPercentByNominal(data.homeowner, data.A)
mc_table = TargetPercentByNominal(data.married_couple, data.A)

nTotal = len(data)

#Q2.a) The class probabilities 

crossTable = pd.crosstab(index = data['A'], columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = (crossTable['Count'] / nTotal)
crossTable = crossTable.drop(columns = ['All'])
print(crossTable)

prob_arr = []

predictor_values = numpy.array([[1, 0, 0],
                      [1, 0, 1],
                      [1, 1, 0],
                      [1, 1, 1],
                      [2, 0, 0],
                      [2, 0, 1],
                      [2, 1, 0],
                      [2, 1, 1],
                      [3, 0, 0],
                      [3, 0, 1],
                      [3, 1, 0],
                      [3, 1, 1],
                      [4, 0, 0],
                      [4, 0, 1],
                      [4, 1, 0],
                      [4, 1, 1]])

for i,j,k in predictor_values:
    print('For predictors: ', i, j, k)
    a0 = gs_table[i].iloc[0] * ho_table[j].iloc[0] * mc_table[k].iloc[0] * crossTable['Percent'].iloc[0]
    a1 = gs_table[i].iloc[1] * ho_table[j].iloc[1] * mc_table[k].iloc[1] * crossTable['Percent'].iloc[1]
    a2 = gs_table[i].iloc[2] * ho_table[j].iloc[2] * mc_table[k].iloc[2] * crossTable['Percent'].iloc[2]
    tot = a0+a1+a2
    a0 = a0/tot
    print('pr(A = 0) =', round(a0,5))
    a1 = a1/tot
    prob_arr.append(a1)
    print('pr(A = 1) =', round(a1,5))
    a2 = a2/tot
    print('pr(A = 2) =', round(a2,5))
    
    print()

max_prob = 0
for i in range(len(prob_arr)):
    if prob_arr[i] >= max_prob:
        max_prob = prob_arr[i]
        max_prob_index = i

print('Predictors that result in the max prob for A = 1 are: ', predictor_values[max_prob_index])

