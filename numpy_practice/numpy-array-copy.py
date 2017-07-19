import numpy as np

#利用arange建立連續的數組  [0,1,2,3] 
a_array = np.arange(4)
print("a_array:",a_array)
b = a_array
c = a_array
d = b
#進行修改
a_array[0]=11
print("修改後的a_array:",a_array)
print("b是否是a_array:", b is a_array)
print("c是否是a_array:", c is a_array)
print("d是否是a_array:", d is a_array)
#修改d
d[1:3]=[22,33]
print("d:", d ," a_array:",a_array)

#deep copy 僅是複製原陣列的值
b = a_array.copy()
a_array[3]=44
print("b:", b ," a_array:",a_array)