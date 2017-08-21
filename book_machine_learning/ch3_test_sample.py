import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sample_moudle.visual.plot_decision_regions import plot_decision_regions

'''
線性可分的資料
'''

#讀取資料
    #使用sklearn的dataset
    #透過iris.target取出y  並且取出前100筆資料
    #target_name=['setosa' 'versicolor' 'virginica']  因此對應y的0,1,2    將setosa 轉為-1  其餘的轉為1   (由於y只取0:100  因此只有setosa與versicolor兩種)
    #feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']   取出其中的petal length (cm)跟 petal width (cm) 作為feature
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

#使用sklearn的cross_validation模組的train_test_split函數  將原始訓練資料切割為訓練集與測試集  (將資料切成30%測試 70%訓練)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#使用sklearn的preprocessing模組的StandardScaler來做標準化 (標準化:讓特徵值滿足標準常態分配 Standard normal distribution 並且每個特徵的平均值都是0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


'''
邏輯斯回歸(LogisticRegression)
'''
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y = y_train, clf=lr, X_highlight = X_test_std)
plt.xlabel('petal length [cm] [Standardized]')
plt.ylabel('pepal width [cm] [Standardized]')
plt.title('LogisticRegression')
plt.legend(loc='upper left')
plt.show()
print('預測各類別的機率(predict_proba):',lr.predict_proba(X_test_std[0,:]))
'''
邏輯斯回歸(LogisticRegression) c值(正規化參數的倒數)不同時對權重的影響
'''
weights, params = [], []
for c in np.arange(-5,5, dtype=int):
    lr = LogisticRegression(C=10**np.int(c), random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**np.int(c))
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params,weights[:,1], linestyle='--', label='petal width')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.title('LogisticRegression with different C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
'''
SVM
'''
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y = y_train, clf=svm, X_highlight = X_test_std)
plt.xlabel('petal length [cm] [Standardized]')
plt.ylabel('pepal width [cm] [Standardized]')
plt.title('SVM')
plt.legend(loc='upper left')
plt.show()

'''
非線性可分的資料
'''

#產生Xor的資料
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1], c='r', marker='s', label='-1')
plt.title('XOR data')
plt.ylim(-3.0)
plt.legend()
plt.show()

'''
svm fit xor
'''
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X=X_xor, y = y_xor, clf=svm)
plt.title('SVM XOR')
plt.legend(loc='upper left')
plt.show()


'''
svm 比較不同 gamma畫出的決策邊界
'''
fig, ax =plt.subplots(1,2, figsize=(8,4))
svm1 = SVC(kernel='rbf', random_state=0, gamma=0.20, C=1.0)
svm1.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y = y_train, clf=svm1, X_highlight = X_test_std, ax=ax[0])
ax[0].set_title('svm gamma=0.2')
ax[0].set_xlabel('petal length [cm] [Standardized]')
ax[0].set_ylabel('pepal width [cm] [Standardized]')
svm2 = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm2.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y = y_train, clf=svm2, X_highlight = X_test_std, ax=ax[1])
ax[1].set_title('svm gamma=100.0')
ax[1].set_xlabel('petal length [cm] [Standardized]')
ax[1].set_ylabel('pepal width [cm] [Standardized]')
plt.show()


'''
決策樹
'''
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
plot_decision_regions(X=X_train, y = y_train, clf=tree, X_highlight = X_test)
plt.xlabel('petal length [cm] ')
plt.ylabel('pepal width [cm] ')
plt.title('Decision Tree')
plt.legend(loc='upper left')
plt.show()

'''
隨機森林
'''
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X=X_train, y = y_train, clf=forest, X_highlight = X_test)
plt.xlabel('petal length [cm] ')
plt.ylabel('pepal width [cm] ')
plt.title('Random Forest')
plt.legend(loc='upper left')
plt.show()

'''
KNN
'''
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X=X_train_std, y = y_train, clf=knn, X_highlight = X_test_std)
plt.xlabel('petal length [cm] [Standardized]')
plt.ylabel('pepal width [cm] [Standardized]')
plt.title('Decision Tree')
plt.legend(loc='upper left')
plt.show()