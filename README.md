
# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed libraries.
2. Read the Placement_data.csv file.And load the dataset.
3. Check the null values and duplicate values.
4. train and test the predicted value using logistic regression
5. calculate confusion_matrix,accuracy,classification_matrix and predict.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Thrinesh Royal.T
RegisterNumber:  212223230226
*/
```
## Head
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/58f776d2-6a40-46ab-8d9c-afab7d7a65fa)

## Copy
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/22b1b092-72aa-4f87-adea-df33b366fd28)

## Null Duplicated
```
data1.isnull().sum()
data1.duplicated().sum()
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/cc75f313-c0f8-4d6f-934d-743e32037183)
## Label Encoder
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print(data1)
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/1b56f9c7-368e-4a7b-96ca-120db946f771)
## Dependent Value
```
y=data1["status"]
y
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/905384b3-5469-497b-a66c-0d73a01493d0)

## Logistic Regression,accuracy,confusion_matrix
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Array:\n",confusion)

```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/09dc2a0f-e9b5-4200-9b58-4f04a448e089)
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/f700dcd8-ecbc-4943-a0ca-7ea94f44e654)
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/d8b9adb2-5d3a-41fc-8fa2-228d32846f43)
## Classification_Matrix,Lr:
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742402/9a1238e1-e037-484a-b9b6-d9cab198aa77)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
