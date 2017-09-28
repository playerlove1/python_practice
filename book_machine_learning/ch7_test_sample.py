from scipy.misc import comb

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

from itertools import product

from sample_moudle.algorithm.ch7.MajorityVoteClassifier import MajorityVoteClassifier
'''
majority voting 多數決

'''

#機率密度函數
def ensemble_error(n_classifier, error):
    #math.ceil 無條件進位
    k_start = int(math.ceil(n_classifier / 2.0))
    #   comb -> c n_classifier取k  二項分配式 error^k * (1-error)^n_classifier-k   針對k由 k_start 到n_classifier + 1 進行運算 得到list
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)

print(ensemble_error(n_classifier=11, error=0.25))
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


#np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6] )
# 0:0.2+0.2
# 1:0.6
#0.6>0.4 因此權重最大的 index值為1
print('bincount:',np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))
print(np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])))

ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
# [0.9*0.2+0.8*0.2+0.4*0.6 , 0.1*0.2+0.2*0.2+0.6*0.6] =[0.58,0.42]
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p)
#0.58>0.42 因此權重最大的 index值為0
print(np.argmax(p))

#載入資料
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
#切割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

#三種演算法

#羅吉斯回歸
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
#決策樹
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
#KNN
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')


pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels): 
    #cross_val_score 回傳每折的分數 成一個list
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

print('加入MV後')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    
#繪製各分類器之ROC
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()
plt.show()


#對資料使用標準化(正規化) 再進行訓練  (為了讓決策樹畫出來的scale與其他兩個演算法一致  因為另外兩個演算法透過pipeline完成了標準化)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

all_clf = [pipe1, clf2, pipe3, mv_clf]

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

#x,y之網格
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    #np.c_ 結合兩個不同的nparray  ex. np.c_([1,2,3],[4,5,6]) = [[1,4],[2,5],[3,6]]
    #ndarray .ravel() =>合併為1維 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #'網格瑱色 按照Z的結果
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    #第一種類別
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], 
                                  X_train_std[y_train==0, 1], 
                                  c='blue', 
                                  marker='^',
                                  s=50)
    #第二種類別
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], 
                                  X_train_std[y_train==1, 1], 
                                  c='red', 
                                  marker='o',
                                  s=50)
    
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5, 
         s='Sepal width [standardized]', 
         ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, 
         s='Petal length [standardized]', 
         ha='center', va='center', 
         fontsize=12, rotation=90)

plt.tight_layout()
plt.show()

#取得ensemble方法中各分類器的參數
print(mv_clf.get_params())

#利用網格搜尋找到最佳的參數
params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

cv_keys = ('mean_test_score', 'std_test_score','params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r" % (grid.cv_results_[cv_keys[0]][r], grid.cv_results_[cv_keys[1]][r] / 2.0, grid.cv_results_[cv_keys[2]][r]))


print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

#取得網格搜尋的最佳參數
mv_clf = grid.best_estimator_
mv_clf.set_params(**grid.best_estimator_.get_params())

#讀取UCI葡萄酒的資料集
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

#所有資料的欄位
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# 只考慮類別2和類別3 所以移除類別標籤為1的資料 
df_wine = df_wine[df_wine['Class label'] != 1]

y = df_wine['Class label'].values
#特徵欄位的部分只考慮 酒精(Alcohol)與色調(Hue)
X = df_wine[['Alcohol', 'Hue']].values

#將類別標籤編碼成二元的形式
le = LabelEncoder()
y = le.fit_transform(y)
#切割訓練與測試資料資料
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.40, random_state=1)

#決策樹 (供bagging使用)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
#利用scikit-learn的 BaggingClassifier 使用500顆未修剪過的決策樹(每顆樹使用不同的bootstrap sample)來形成一個ensemble方法
bag = BaggingClassifier(base_estimator=tree, n_estimators=500,  max_samples=1.0,  max_features=1.0,  bootstrap=True,  bootstrap_features=False,  n_jobs=1, random_state=1)

#比較單一棵決策樹與一個ensemble方法的準確度差異

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))

#根據輸出結果可以得知 兩者對於訓練資料的預測都是完全正確的  但對於測試資料的準確度卻有不同(ensemble >單一決策樹)  因此單一決策樹可能發生了overfiting

#作圖比較單一決策樹 與ensemble方法的決策邊界差異
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

#網格
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))


for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    #網格填色(背景)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    
    #第一個類別的資料
    axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
    
    #第二個類別的資料
    axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', marker='o')

    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

#利用適應強弱來提升弱學習器的效能
'''
特殊的ensemble方法:強化法(boosting)  

ensemble是利用許多非常簡單的弱分類器(weak learner) (每個weak learner只比隨機猜測好一些)
讓弱學習器 能從誤判的訓練樣本中 去學習  藉此提高ensemble方法的效能 
不同於bagging 對訓練資料集是採用  "不放回式隨機抽樣"  來製作出訓練子集 並使用 隨機猜測 來做預測

原始的強化法(boosting)可以歸納如下:
1. 由訓練資料集D中以 "不放回式隨機抽樣" 產生樣本子集 d1 以d1來訓練分類器c1
2. 重複1的步驟直到所有分類器都訓練過了
3. 結合利用這些訓練過的分類器 以多數決的方式 代表預測結果

AdaBoost(Adaptive Boosting)
使用完整的訓練資料集來訓練弱分類器
每次迭代的過程中 訓練樣本會從上一輪 方法中錯誤的預測 重新給定加權值(初始每個樣本為等值) 使其成為更好更強的分類器

1. 初始化向量w 為相等加權值(對每個樣本而言)  sum(w)=1
2. 對每次強化中的分類器j 做以下步驟
    Cj=train(X,y,w)
    y_hat=predict(Cj,X)
    epsilon = w . (y_hat==y)
    alpha j =0.5 log(1-epsilon/epsilon)
    w := w x exp(-alpha j x y_hat x y)
    1:w := w/ sigma i w i
3. 計算最後預測(集合每個分類器的alpha j加權結果) 
'''
#決策樹(供adaboost使用)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
#使用scikit-learn 訓練AdaBoost ensemble分類器 以500棵單層決策樹訓練
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

#比較單棵決策樹與Adaboost的決策邊界差異
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))


for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
   #網格填色(背景)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()