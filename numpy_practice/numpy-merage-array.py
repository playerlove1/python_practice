import numpy as np


a_array = np.array([1,1,1])

b_array = np.array([2,2,2])
print("a_array:",a_array," b_array",b_array)
#利用np.vstack將兩個矩陣以新增列的方式加入 (行數要一樣)
print("vstack:",np.vstack((a_array,b_array)))
#利用np.hstack將兩個矩陣以新增行的方式加入 (列數要一樣)
print("hstack:",np.hstack((a_array,b_array)))

a_array=a_array.reshape((3,1))
b_array=b_array.reshape((3,1))
print("轉換後  a_array:",a_array," b_array",b_array)
#利用np.hstack將兩個矩陣以新增行的方式加入 (列數要一樣)
print("hstack:",np.hstack((a_array,b_array)))

#利用np.concatenate結合多個矩陣  axis=1代表 以行為基準合併矩陣
result=np.concatenate((a_array,b_array,a_array),axis=1)
print(result)