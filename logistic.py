import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import statsmodels.api as api


data = pd.read_csv('/Users/snehamitta/Desktop/ML/Assignment3/Purchase_Likelihood.csv')

#Q2.c)
Aa = data['A'].astype('category')
y = Aa
y_category = y.cat.categories

p1 = data[['group_size']].astype('category')
X = pd.get_dummies(p1)
X = X.join(data[['homeowner','married_couple']])
X = api.add_constant(X, prepend=True)

logit = api.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-6)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

#Q2.f)
x1 = pd.DataFrame(columns = ['const', 'group_size_1', 'group_size_2', 'group_size_3', 'group_size_4', 'homeowner', 'married_couple'])
x1.loc[0] = [1.0,0,1,0,0,1,1]
y_predProb = thisFit.predict(x1)
print(y_predProb)

#Q2.g)
y_predProb1 = thisFit.predict(X)
print(y_predProb1.loc[y_predProb1[0].idxmax()])
print(data.iloc[569830])


