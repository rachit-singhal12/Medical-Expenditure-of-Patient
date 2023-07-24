import pandas as pd
data = pd.read_csv('Medical expenditure of patient\\medical_cost_data\\insurance.csv')

#preprocess the data
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()

data['sex']=le1.fit_transform(data['sex'])
data['region']=le1.fit_transform(data['region'])
data['smoker']=le1.fit_transform(data['smoker'])

#splitting the data
x = data.iloc[:,0:6]
y = data['charges']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest  = train_test_split(x,y)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
xtrain = mms.fit_transform(xtrain)
xtest = mms.fit_transform(xtest)

#models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
dtc = DecisionTreeRegressor()
svc = SVR()
rfr = RandomForestRegressor()

lr.fit(xtrain,ytrain)
dtc.fit(xtrain,ytrain)
svc.fit(xtrain,ytrain)
rfr.fit(xtrain,ytrain)

model1 = lr.predict(xtest)
model2 = dtc.predict(xtest)
model3 = svc.predict(xtest)
model4 = rfr.predict(xtest)

import math
from sklearn.metrics import mean_squared_error,accuracy_score
print("Linear regression",math.sqrt(mean_squared_error(model1,ytest)))
print("Decision tree regression",math.sqrt(mean_squared_error(model2,ytest)))
print("support vector regression",math.sqrt(mean_squared_error(model3,ytest)))
print("random forest regression",math.sqrt(mean_squared_error(model4,ytest)))