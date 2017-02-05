#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

#訓練集資料減量以換取速度 (減量到原先的1%)

#features_train = features_train[:(len(features_train)//100)]
#labels_train = labels_train[:(len(labels_train)//100)]



#########################################################
### your code goes here ###


#引入函式庫
from sklearn.svm import SVC
#選擇kernel 初始化分類器

#linear kernel
#clf = SVC(kernel="linear")

#rbf kernel  不同c值
# clf = SVC(kernel="rbf",C=10)
# clf = SVC(kernel="rbf",C=100)
# clf = SVC(kernel="rbf",C=1000)
clf = SVC(kernel="rbf",C=10000)

#給定訓練資料  特徵 與 標籤
clf.fit(features_train, labels_train)


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
print("精準度(accuracy):",accuracy_score(pred, labels_test))

print("預測結果:","10:",pred[10],"\n26:",pred[26],"\n50:",pred[50])


from collections import Counter
c = Counter(pred)
print ("Chris(1)的郵件數:", c[1])

#########################################################

# default 設定  (完整資料 linear kernel)
# 訓練時間: 147.241 秒(s)
# 預測時間: 15.354 秒(s)
# 精準度(accuracy): 0.984072810011


# 資料減量至1% (linear kernel)
# 訓練時間: 0.081 秒(s)
# 預測時間: 0.869 秒(s)
# 精準度(accuracy): 0.884527872582

# 資料減量至1% (rbf kernel)
# 訓練時間: 0.092 秒(s)
# 預測時間: 0.996 秒(s)
# 精準度(accuracy): 0.616040955631



# 資料減量至1% (rbf kernel)  c=10
# 訓練時間: 0.092 秒(s)
# 預測時間: 0.983 秒(s)
# 精準度(accuracy): 0.616040955631

# 資料減量至1% (rbf kernel)  c=100
# 訓練時間: 0.093 秒(s)
# 預測時間: 0.986 秒(s)
# 精準度(accuracy): 0.616040955631

# 資料減量至1% (rbf kernel)  c=1000
# 訓練時間: 0.09 秒(s)
# 預測時間: 0.947 秒(s)
# 精準度(accuracy): 0.821387940842

# 資料減量至1% (rbf kernel)  c=10000
# 訓練時間: 0.087 秒(s)
# 預測時間: 0.783 秒(s)
# 精準度(accuracy): 0.892491467577

# 完整資料 (rbf kernel)  c=10000
# 訓練時間: 96.364 秒(s)
# 預測時間: 9.861 秒(s)
# 精準度(accuracy): 0.990898748578