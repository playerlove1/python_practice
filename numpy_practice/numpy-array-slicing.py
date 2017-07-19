import numpy as np

#利用arange建立連續的數組  [0,1,2,3,4,5,6,7,8,9,10,11] 並reshape
a_array = np.arange(12).reshape((3,4))
print("a_array:",a_array)
#利用np.split(原陣列,切成幾份(需可整除),axis=按 列切or行切) 進行等項分割
print("等項切割 ",np.split(a_array,2,axis=1))
#利用np.array_split(原陣列,切成幾份,axis=按 列切or行切) 進行不等項分割
print("不等項切割 ",np.array_split(a_array,3,axis=1))

#按照列切割
print("vsplit",np.vsplit(a_array,3))
#按照行切割
print("hsplit",np.hsplit(a_array,2))

