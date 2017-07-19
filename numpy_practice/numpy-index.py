import numpy as np

#利用arange建立連續的數組 [3,4,5,6,7,8,9,10,11,12,13,14]
sample_array = np.arange(3,15)
print("原始矩陣:",sample_array)
#索引值為3的 為 6  (索引值由0開始計算)
print("索引值為3的元素",sample_array[3])

#轉為3列4行的矩陣
sample_array = sample_array.reshape((3,4))
print("轉換為3列4行後:",sample_array)
#第3列第4行的元素  (索引值由0開始計算)
print("第3列第4行的元素",sample_array[2][3])
#第2列的所有元素
print("第2列的所有元素",sample_array[1,:])
#第2行的所有元素
print("第2行的所有元素:",sample_array[:,1])

#用for輸出矩陣

#每一列
print("依序輸出每一列")
for row in sample_array:
    print(row)
#每一行(利用原矩陣的轉置達成目的)
print("依序輸出每一行")
for column in sample_array.T:
    print(column)

#所有元素  (利用  矩陣.flat這個迭代器歷遍所有元素)
print("依序輸出每個元素")
for item in sample_array.flat:
    print(item)