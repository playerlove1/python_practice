#參考自:https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/2-2-np-array/
import numpy as np


#利用dtype指定陣列內的資料型態
sample_array=np.array([2,3,4],dtype=np.int)
print(sample_array.dtype)

#建立元素全為0的陣列 3列 4行 的陣列
zero_array=np.zeros((3,4))
print(zero_array)

#建立全為1的陣列  3列 4行 的陣列
one_array=np.ones((3,4),dtype=np.int)
print(one_array)


#利用arange建立連續的數組  ex 10,12,14,16,18 (由10~20 每隔2挑一次) 為一個一維陣列
continuous_array=np.arange(10,20,2) 
print(continuous_array)

#利用arange建立連續的數組 		1列  12行 的陣列
reshape_array=np.arange(12)
print(reshape_array)
#利用reshape改變陣列的行列值   改為 3列 4行
reshape_array=reshape_array.reshape(3,4)
print(reshape_array)

#利用linspace 由 1~20 切為20等段的資料
linspace_array=np.linspace(1,20,20)
print(linspace_array)

