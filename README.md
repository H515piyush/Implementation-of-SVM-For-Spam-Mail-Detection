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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = svm.SVC (kernel='linear') 
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score (y_test, predictions)) 
print("Classification Report: ")
print(classification_report (y_test, predictions))

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
