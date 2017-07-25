import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sample_moudle.algorithm.ch4.SBS import SBS

#數值型資料
csv_data='''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            0.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print("原始Dataframe(數值型):\n",df)
print("各欄位(行)的缺失值個數:\n",df.isnull().sum())

df_drop =df.dropna(axis=1)
print("移除缺失行後的Dataframe(數值型):\n",df_drop)

#利用sklearn的填補模組進行缺失值填補 (median 中位數  most_frequent 眾數)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print("填補後的Dataframe:\n",imputed_data)

#類別型資料
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],\
                    ['red', 'L', 13.5, 'class2'],\
                    ['blue', 'XL', 15.3, 'class1']
                    ])
df.columns = ['color', 'size', 'price', 'classlabel']
print("原始Dataframe(類別型):\n",df)
#假設知道各尺寸間的大小倍率
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print("size mapping後的Dataframe(類別型):\n",df)
#做出類別的mapping
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print("class_mapping:",class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print("class_mapping後的Dataframe(類別型):\n",df)
#反類別的mapping
inv_class_mapping ={v: k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print("反class_mapping後的Dataframe(類別型):\n",df)

#利用sklearn.preprocessing的LabelEncoder 完成類別的mapping
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print("class mapping後:",y)
y = class_le.inverse_transform(y)
print("inverse class mapping後:",y)

#將顏色轉換為數值  (但此種轉法會使得顏色 產生大小間的關係  但實際上並不是)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print("直接轉數值:\n",X)

#為了避免類別屬性轉成數值會產生大小關係  會使用 one-hot encoding  也就是 dummy feaute (進行虛擬變數的設計)
#使用sklearn.preprocessing的 OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0])
#預設回傳為稀疏矩陣  可以利用toarray 轉換為Numpy的陣列  (也可在初始化ohe時代入參數 sparse=False 則不需要使用 toarray)
X_ohe=ohe.fit_transform(X).toarray()
print("one hot encoding:\n",X_ohe)

#使用pandas 的get_dummies方法進行虛擬變數的轉換
X_pd_dummies = pd.get_dummies(df[['price', 'color', 'size']])
print("利用pandas的get_dummies方法:\n",X_pd_dummies)

#
uci_wine_url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(uci_wine_url, header=None)
df_wine.columns= ['Class label','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',\
            'Magnesium', 'Total phenols', 'Flavanoids','Nonflavanoid phenols',\
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',\
            'Proline']

print(df_wine.head())

X, y = df_wine.iloc[:,1:].values ,df_wine.iloc[:,0].values
#使用sklearn的cross_validation模組的train_test_split函數  將原始訓練資料切割為訓練集與測試集  (將資料切成30%測試 70%訓練)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#利用sklearn.preprocessing的MinMaxScaler模組進行 特徵縮放  (normalization)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)
#使用sklearn的preprocessing模組的StandardScaler來做標準化 (標準化:讓特徵值滿足標準常態分配 Standard normal distribution 並且每個特徵的平均值都是0)
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

print('截距:',lr.intercept_)
print('lr.coef_:',lr.coef_)


#l1 正規化
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',\
          'black', 'pink', 'lightgreen', 'lightblue', 'gray',\
          'indigo','orange']

weights, params= [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**np.int(c), random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**np.int(c))

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors): 
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color = color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='lower left', fancybox=True)
plt.show()

#
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
knn.fit(X_train_std, y_train)
print('Test accuracy:', knn.score(X_test_std, y_test))

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

sfm = SelectFromModel(forest, threshold=0.15, prefit=True)
X_selected = sfm.transform(X_train)

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

