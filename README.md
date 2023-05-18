# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the given dataset.
2.Fitting the dataset into the training set and test set.
3.Applying the feature scaling method.
4.Fitting the logistic regression into the training set.
5.Prediction of the test and result
6.Making the confusion matrix 7.Visualizing the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KARTHICK RAJ M
RegisterNumber: 212221040073
*/


```
# implement a simple regression model for predicting the marks scored by the students

import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![logistic regression using gradient descent](sam.png)

prediction of test result:


![image](https://github.com/KARTHICKRAJM84/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128134963/dc1eca43-a90d-4449-a250-225bb5a6aef7)


Confusion Matrix:

![image](https://github.com/KARTHICKRAJM84/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128134963/9c56e505-731d-45d0-9c3d-3362d04494ac)


Accuracy:

![image](https://github.com/KARTHICKRAJM84/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128134963/49caae4a-379c-4177-a65e-b14fb41d820b)


Recalling Sensitivity and Specificity:


![image](https://github.com/KARTHICKRAJM84/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128134963/f577a839-f8b2-4afc-9f24-2ece698c2d29)



Visulaizing Training set Result:

![image](https://github.com/KARTHICKRAJM84/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128134963/dfa789ad-f201-4da7-9add-ee1897e0d24a)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

