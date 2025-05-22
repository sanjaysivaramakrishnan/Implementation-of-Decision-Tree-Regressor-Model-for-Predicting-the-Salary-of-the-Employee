# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas as pd and import the required dataset.

2.Calculate the null values in the dataset.

3.Import the LabelEncoder from sklearn.preprocessing

4.Convert the string values to numeric values.

5.Import train_test_split from sklearn.model_selection.

6.Assign the train and test dataset.

7.Import DecisionTreeRegressor from sklearn.tree.

8.Import metrics from sklearn.metrics.

9.Calculate the MeanSquareError.

10.Apply the metrics to the dataset.

11.Predict the output for the required values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by   : Sanjay Sivamakrishnan M
RegisterNumber : 212223240151
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\Salary.csv')
df.head()
le = LabelEncoder()
df['Position'] =  le.fit_transform(df['Position'])
df.head()
X = df.drop(columns=['Salary'])
y = df['Salary']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('The mean squre error is : ',mse)
model.predict([[5,6]])

```

## Output:

![image](https://github.com/user-attachments/assets/36d5c132-a3ee-48e1-af98-2893d0cd58d2)
![image](https://github.com/user-attachments/assets/96dabaae-5c4b-4203-83ba-b05b9d08ff0b)
![image](https://github.com/user-attachments/assets/1443f2b2-0abe-4411-b6c8-43616624095f)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
