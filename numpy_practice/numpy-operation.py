import numpy as np


#利用arange建立連續的數組  ex 10,20,30,40 (由10~40 每隔10挑一次) 為一個一維陣列
a_array = np.arange(10,50,10) 
#利用arange建立連續的數組(預設由0開始) 		1列  4行 的陣列  [0,1,2,3]
b_array = np.arange(4)

#矩陣中元素逐個相加
result = a_array + b_array
print("a_array + b_array:",result)
#矩陣中元素逐個相減
result = a_array - b_array
print("a_array - b_array:",result)
#矩陣中元素逐個相乘
result = a_array * b_array
print("a_array * b_array:",result)
#矩陣中元素逐個相除
result = a_array / b_array
print("a_array / b_array:",result)

#矩陣平方
print("a_array的平方:",a_array**2," b_array的平方:",b_array**2)

#取sin值
print("a_array的sin:",np.sin(a_array)," b_array的sin:",np.sin(b_array))
#取cos值
print("a_array的cos:",np.cos(a_array)," b_array的cos:",np.cos(b_array))
#取tan值
print("a_array的tan:",np.tan(a_array)," b_array的tan:",np.tan(b_array))

#找出某arrray中小於某個數的布林陣列  (亦可用> or == or >= or <=)
result=b_array<3
print("b_array中小於3的布林陣列:",result)


#將a_array 與b_array 轉換為2維矩陣
a_array=a_array.reshape((2,2))
b_array=b_array.reshape((2,2))
print("轉換後 a_array:",a_array," b_array:",b_array)

#矩陣乘法 (內積)
result=np.dot(a_array,b_array)
print("矩陣乘法(dot)的結果:",result)

#隨機生成0-1的小數的陣列
result=np.random.random((2,2))
print("隨機生成結果:",result)
#找array中的元素總和,最小值,最大值
print("sum:",np.sum(result)," min:",np.min(result)," max:",np.max(result))
#指定由矩陣的列或行 進行運算  (利用axis)   (直行橫列  因此axis=0為列,axis=1為行)
print("sum:",np.sum(result,axis=0)," min:",np.min(result)," max:",np.max(result))

#平均
print("mean:",np.mean(result)," average:",np.average(result))
#中位數
print("中位數:",np.median(result))

#累加各個元素
print("累加各個元素:",np.cumsum(result))
#與後一項的差
print("與後一項的差:",np.diff(result))
#輸出行列數(第一個array為所在列數  第二個array為所在行數)
print("行列兩個array:",np.nonzero(result))
#排序元素(逐列)
print("排序後:",np.sort(result))
#轉置
print("轉置後:",np.transpose(result))
print(result.T)
#由矩陣中擷取子矩陣 (介於 0.5到1之間的數 保留原值  大於1的視為1 小於0.5的視為0.5)
print("clip:",np.clip(result,0.5,1))