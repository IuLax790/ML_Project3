import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
Credit_India = pd.read_csv("C:\\Information_Science\\My_projects\\Credit card transactions - India - Simple.csv")
print(Credit_India)
X = Credit_India.iloc[:,:-1].values

y = Credit_India.iloc[:, -1].values
print(X)
print(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False)
Z = onehotencoder.fit_transform(X[:,[0]])
X=np.hstack((X[:,:0],Z)).astype('int')
Z = onehotencoder.fit_transform(X[:,[1]])
X=np.hstack((X[:,:1],Z)).astype('int')
Z = onehotencoder.fit_transform(X[:,[2]])
X=np.hstack((X[:,:2],Z)).astype('int')
Z = onehotencoder.fit_transform(X[:,[3]])
X=np.hstack((X[:,:3],Z)).astype('int')
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
import statsmodels.api as sm
import statsmodels.tools.tools as tl
X = tl.add_constant(X)

SL = 0.05
X_opt = X[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

numVars = len(X_opt[0])
for i in range(0,numVars):
    regressor_OLS = sm.OLS(y,X_opt).fit()
    max_var = max(regressor_OLS.pvalues).astype(float)
    if max_var > SL:
        new_Num_Vars = len(X_opt[0])
        for j in range(0,new_Num_Vars):
            if (regressor_OLS.pvalues[j].astype(float)==max_var):
                X_opt = np.delete(X_opt,j,1)
print(regressor_OLS.summary())

print(y_pred)
print(regressor_OLS.pvalues)
print(regressor_OLS.params)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred)
plt.xlabel("Variables")
plt.ylabel('Spending in Ruppees')
plt.title('Spending Prediction')

print(plt.show())
predicted_y = regressor.predict
print(predicted_y)
y_pred = regressor.predict(X_test)
result = pd.DataFrame(y_pred)
result.columns = ["prediction"]
print(result.to_csv("C:\\Information_Science\\My_Projects\\IndiaSpending_Projection.csv"))
