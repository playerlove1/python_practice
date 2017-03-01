#測試正規化
from sklearn import preprocessing
import numpy as np
data=[1,2,2,2,8,10]
print(np.mean(data))
print(np.std(data))
d=preprocessing.MinMaxScaler().fit(data)
print(d.transform(data))
print(preprocessing.MinMaxScaler().fit(data).inverse_transform(d.transform(data)))

# 標準差為1 平均值為0的正規化
# preprocessing.scale(data)
# 將train data的 scale 套用到 test data
# preprocessing.StandardScaler().fit(train data).transform(test data)
# 將資料正規化到[0,1]區間
# preprocessing.MinMaxScaler().fit_transform(data)