#參考自:https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/2-1-np-attributes/
import numpy as np

#由list轉為矩陣
sample_array=np.array([[1,2,3],[2,3,4]])

#取得相關屬性
print ("維度:",sample_array.ndim)
print ("列數與行數:",sample_array.shape)
print ("元素個數:",sample_array.size)

