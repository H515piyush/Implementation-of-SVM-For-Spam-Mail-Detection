# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:

```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:piyush kumar 
RegisterNumber:212223220075  
*/
import chardet
file='/Content/spam.csv'
with open(file, 'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data['v2'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Head() :

![328008742-37e16030-4437-49d8-b3ec-3f3c68a7d5fa](https://github.com/H515piyush/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147472999/bc365f69-6ea9-419f-bc85-d662b6ec0ddd)

Kernel Model:

![328008809-03a2362a-acf7-4f86-b310-309aca2446d6](https://github.com/H515piyush/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147472999/8e1771f1-6e4f-4f1f-8972-145116e958b1)

#Accuracy and Classification Report :

![328008809-03a2362a-acf7-4f86-b310-309aca2446d6](https://github.com/H515piyush/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147472999/1b30def8-5c0b-4cae-a032-eacc707920da)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
