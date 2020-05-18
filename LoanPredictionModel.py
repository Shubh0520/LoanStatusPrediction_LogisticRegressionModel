"""
Created on Mon May 18 09:46:45 2020

@author: Shubham
"""

"""Importing libraries"""
import pandas as pd

"""Reading dataset and creating a copy"""
loandata = pd.read_csv('01Exercise1.csv')
loanprep = loandata.copy()

"""Calculating the sum and checking the missing values"""

missing_count = print(loanprep.isnull().sum(axis=0))

"""Dropping the missing values and rechecking null values"""
loanprep = loanprep.dropna()
recheck_missing_count = print(loanprep.isnull().sum(axis=0))

"""Dropping irrelevant column which will not effect predictions
dropping gender"""

loanprep = loanprep.drop(['gender'], axis=1) #axis=1 for column

"""Creating dummy variables for categorical data
such as Married, Credit history, status"""
loanprep = pd.get_dummies(loanprep, drop_first=True)

"""Normalising the dataset for Income,loanamt"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
loanprep['income'] = scaler.fit_transform(loanprep[['income']])
loanprep['loanamt'] = scaler.fit_transform(loanprep[['loanamt']])

"""Creating X & Y"""
Y = loanprep[['status_Y']]
X = loanprep.drop(['status_Y'], axis=1)

"""Splitting the dataset"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=2222,stratify=Y)

"""Create the Logical regressor"""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)

"""Building confusion matrix and get the accuracy/score"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)

"""For more accuracy we can use score method"""
final_score = lr.score(X_test, Y_test)