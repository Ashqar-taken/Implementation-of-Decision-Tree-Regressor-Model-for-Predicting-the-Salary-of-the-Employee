# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ashqar Ahamed S.T
RegisterNumber: 212224240018
```
```

import pandas as pd
data=pd.read_csv(r"C:\College\SEM 2\Machine Learning\Exp9\Salary.csv")
print("Data:\n",data.head())

print("Data Info:\n",data.info)

print('\n',data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("Data after LabelEncoder:\n",data.head())

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
print("MSE:\n",mse)

r2=metrics.r2_score(y_test,y_pred)
print("R squared:\n",r2)

new_pred = dt.predict([[5,6]])
print("Prediciton for new values:\n",new_pred)
```
## Data:
![data](https://github.com/user-attachments/assets/74f20048-1cfe-4e33-94c8-ed88116d2eeb)

## Data Info:
![image](https://github.com/user-attachments/assets/ead009cb-3df9-4458-8a79-636fa84bbf7b)
## Null values:
![image](https://github.com/user-attachments/assets/313f04f0-ea46-4a01-a095-5c6f0014c5db)
## Data After LabelEncoder:
![image](https://github.com/user-attachments/assets/0f3f80a4-d45e-4160-88c6-d81116f023c5)
## Mean Squared Error:
![image](https://github.com/user-attachments/assets/af0f2fb0-848f-43af-9fb1-35c2981ebbe5)
## R squared:
![image](https://github.com/user-attachments/assets/dc21a970-31ee-4747-b1f0-c13dee0996f0)
## Prediciton for New value:
![image](https://github.com/user-attachments/assets/e4da61af-ea95-4f7b-a3a0-851b2380950d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
