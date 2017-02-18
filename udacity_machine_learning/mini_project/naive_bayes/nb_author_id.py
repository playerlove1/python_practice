#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



print(type(features_train))

#########################################################
### your code goes here ###

#引入函式庫
from sklearn.naive_bayes import GaussianNB
#初始化分類器
clf = GaussianNB()

#開始訓練的時間
start_train_time=time()

#給定訓練資料  特徵 與 標籤 (訓練階段)
clf.fit(features_train, labels_train)
print("訓練時間:",round(time()-start_train_time,3),"秒(s)")


#開始預測的時間
start_pred_time=time()

#預測值
pred = clf.predict(features_test)
print("預測時間:",round(time()-start_pred_time,3),"秒(s)")

#取得預測精準度
from sklearn.metrics import accuracy_score
print ("精準度(accuracy):",accuracy_score(pred, labels_test))

#########################################################


