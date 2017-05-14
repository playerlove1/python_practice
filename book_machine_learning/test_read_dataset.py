#測試資料集查看



#方法一:使用sklearn的dataset

from sklearn import datasets
#回傳的為 Dictionary-like的物件     屬性:data,target,target_names,feature_names,  DESCR(資料集的簡述)
iris = datasets.load_iris()

#整份資料集的簡述
print(iris.DESCR)

print(iris)
print(iris.target)
print(iris.target_names)
print(iris.feature_names)

print('--------------------------------')
#方法二:使用pandas 載入UCI ML Repository
import pandas as pd

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

print(df.tail())
