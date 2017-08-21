import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score

from scipy import interp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline



#威斯康辛乳癌dataset
#共32個特徵 569個樣本  M=惡性 B=良性(index=1)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
print('rows, columns:', df.shape)
print(df.head())

#取3-32特徵
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
#將原先的M,B標籤轉換為1,0
le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B'])
#切割訓練資料集80%與測試資料集20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

#利用Pipeline 將 標準化 pca 分類器  等串聯在一起  (詳細如ch6_pipeline原理所示)
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
#會將dataset 先經由 StandardScaler fit_transform 再將輸出 交給PCA 作維度縮減  再將輸出 交給 羅吉斯回歸 做訓練
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
y_pred = pipe_lr.predict(X_test)

#在建立機器學習模型的過程中關鍵的步驟 「估計」模型在 「沒見過的資料」的預測能力
#交叉驗證技術 : 保留交叉驗證法(holdout cross-vaildation) 、 k折交叉驗證法(k-fold cross-vaildation)
'''
保留交叉驗證法(holdout cross-vaildation)
    將資料拆分成訓練資料與測試資料 (藉此調整參數 比較不同參數的組合 提高對未知資料的預測能力  此過程稱為模型選擇 model selection)
    實務上會將資料拆成 三種資料集:訓練資料集(training set)、驗證資料集(vaildation set)、測試資料集(test set)
    在訓練跟驗證階段 做模型選擇  再用測試集估計  一般化誤差
    
    缺點: 如何拆分資料集是相當敏感的 會使得效能差異非常大

k折交叉驗證法(k-fold cross-vaildation)
    隨機將訓練資料集分割成k折，其中樣本不放回(即每筆資料只會在其中1折出現)。
    利用其中的k-1折來進行模型訓練  剩下的那1折來進行 測試資料集
    將上述步驟重複k次就會得到k個模型 與 此k個模型的效能估計
    利用k折交叉驗證找到適合的參數 當參數校正完成後 再使用獨立的測試資料集 進行效能評估
    
    其中之變形 留一交叉驗證 (Leave-One-Out,LOO)
    將k設定為樣本數，因此 每次只會有一筆訓練資料來做驗證  再只有一個「非常小」的訓練集的時候會採用
    另一改良 分層k折交叉驗證(stratified k-fold cross-vaildation)
    讓每折資料中  類別大小的比例 跟原始訓練資料集的比例是相同的
'''
#使用分層k折交叉驗證
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#利用 cross_val_score 可以利用不同的CPU 分散計算 加速
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#利用驗證取曲線來挑適合的參數

#建立pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty='l2', random_state=0))])
#學習曲線預設會採用 分層k折交叉驗證 來計算交叉驗證的準確性 利用CV 不同大小的訓練集來取得訓練分數與測試分數
train_sizes, train_scores, test_scores =learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
#使用fill_between的函數 在圖形中 加入 平均正確率與標準差 來描述 估計值的變異數
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()

#利用驗證曲線討論不同lr的C之結果
#提高正規化的強度 (較小的c)
#降低正規化的強度 (較大的c)
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve( estimator=pipe_lr, X=X_train, y=y_train, param_name='clf__C', param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()

#利用網格搜尋來調整參數
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
#對於線性的SVM只調C ,RBF 調C 跟 gamma
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range,  'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

#利用網格搜尋的結果取得最佳參數的模型
clf = gs.best_estimator_
#利用該模型 fit 訓練資料
clf.fit(X_train, y_train)
#用測試資料集檢驗成效
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

#巢狀交叉驗證(nested cross-vaildation) 架構如ch6_nested.png
#又稱為 外折x內折  交叉驗證  (5x2 交叉驗證)

#內折 cv2
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)

# Note: Optionally, you could use cv=2 
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.
#外折 cv5
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


#使用巢狀交叉驗證 於決策樹
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#由上面SVM與決策樹的的比較可以發現  經參數挑選後 前者的分類能力是較好的

'''

其他不同的效能指標

預測錯誤率(error,ERR) 預測正確率(accuracy,ACC)
ERR =分類錯誤/所有樣本 ACC=分類正確/所有樣本=1-ERR

真陽率(True positive rate,TPR)與假陽率(False positive rate,FPR)  處理不平衡類別問題的 效能指標
TPR=實際為真且預測為真/實際為真   FPR = 實際為假卻預測為真/實際為假  
以腫瘤為例，減少良性被誤判為惡性 (假陽率 FPR)

精確度(precision,PRE) 召回率(recall,REC)=真陽率TPR
PRE = 實際為真且預測為真/預測為真  REC=TPR=實際為真且預測為真/實際為真
通常使用 精確度 與 召回率 的組合 稱為F1分數
F1=2*(PRE*REC)/(PRE+REC)

'''

#混淆矩陣(confusion matrix) TP FN FP TN
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#劃出混淆矩陣
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

#使用 sklearn的metrics模組查看相關指標
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#預設是使用標籤為1的當正類  可以透過make_scorer建立自己的計分器(scorer)
scorer = make_scorer(f1_score, pos_label=0)
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'clf__C': c_gamma_range, 'clf__kernel': ['linear']}, {'clf__C': c_gamma_range, 'clf__gamma': c_gamma_range, 'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10, n_jobs=1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

#接收操作特徵圖 (Receiver operating characteristic,ROC)
'''
ROC圖是選擇分類模型的有用工具，可以基於模型效能(假陽率,真陽率)來選擇
其對角線可以被解釋成隨機猜測  低於對角線 表示比隨機猜還差的模型
好的ROC模型應該要落在真陽率1 假陽率為0的部分
'''
#繪製ROC曲線圖
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(penalty='l2', random_state=0, C=100.0))])

X_train2 = X_train[:, [4, 14]]


cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

#曲線下面積(Area under curve,AUC)


pipe_lr = pipe_lr.fit(X_train2, y_train)
y_labels = pipe_lr.predict(X_test[:, [4, 14]])
y_probas = pipe_lr.predict_proba(X_test[:, [4, 14]])[:, 1]

print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_probas))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_labels))

#加權宏觀平均
pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True,  average='micro')