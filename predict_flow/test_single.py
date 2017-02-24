#單一個VDS資料的測試
import os
import numpy as np
from tools import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP


# 日期字串
# time_str="7/10/2006 6:00"
# 引用datetime
# from datetime import *
# 轉為日期
# dt=datetime.strptime(time_str, "%m/%d/%Y %H:%M")
# 將datime轉為日期
# dt.date()
# 取小時
# dt.hour
# 取分
# dt.hour



#目前的目錄
root_dir=os.getcwd()

#放檔案的目錄路徑
data_dir="data"+"\\"+"400300"+"\\"+"train"+"\\"


#data目錄下各資料夾的list
data_dir_list=os.listdir(data_dir)
#資料的名稱

data_name_xls="pems_output.xls"

#所有資料的dataframe 陣列
file_dfs=[]

for dir in data_dir_list:
    file_excel=get_time_flow(read_xls_file(data_dir+dir+"\\"+data_name_xls,'Report Data'))
    file_dfs.append(file_excel.iloc[:,1])

    
#格式化資料
train_data=format_data_combine(5,1,file_dfs)
# train_data=format_data(5,1,file_dfs[4])



#取出feature與target 並轉為np array 提供機器學習演算法運算
train_feature=(train_data.iloc[:,0:8]).as_matrix(columns=None)
train_target=(train_data.iloc[:,8]).as_matrix(columns=None)
# train_feature=(train_data.iloc[:,0:4]).as_matrix(columns=None)
# train_target=(train_data.iloc[:,4]).as_matrix(columns=None)


#呼叫SVR演算法並指定參數
clf = SVR(kernel="rbf",C=100,gamma=0.0001)
#針對演算法 使用訓練資料fit (即訓練的意思)
clf.fit(train_feature,train_target)
#使用已經訓練過的模型進行預測原來的特徵資料

params=clf.get_params()
print(params)

y_predict=clf.predict(train_feature)

#將結果化成圖

x=np.arange(len(train_target))
draw_picture(x,train_target,y_predict)


#用測試資料 並計算結果
data_dir_test="data"+"\\"+"400300"+"\\"+"test"+"\\"+"0523-0527"

test_dfs=[]
test_dfs.append(file_dfs[1])
test_dfs.append(file_dfs[2])
test_dfs.append(file_dfs[3])
test_dfs.append(get_time_flow(read_xls_file(data_dir_test+"\\"+data_name_xls,'Report Data')).iloc[:,1])

test_data=format_data_combine(5,1,test_dfs)
# test_data=format_data(5,1,test_dfs[3])

test_feature=(test_data.iloc[:,0:8]).as_matrix(columns=None)
test_target=(test_data.iloc[:,8]).as_matrix(columns=None)
# test_feature=(test_data.iloc[:,0:4]).as_matrix(columns=None)
# test_target=(test_data.iloc[:,4]).as_matrix(columns=None)

test_y_predict=clf.predict(test_feature)

#衡量結果之公式 
#MAPE sigma( (實際值-預測值)/ 實際值  *100 )/資料筆數
MAPE=np.sum((abs(test_target-test_y_predict)/test_target*100))/len(test_target)
print("MAPE:",MAPE)


print("score:",clf.score(test_feature, test_target))

#AIC = log(e mle)+ 2d/n
#n:資料筆數 d:參數個數  e mle=sigma((實際值-預測值)^2/資料筆數)
#BIC=log(e mle)+ d log(n)/n




#演算法
#NN (類神經網路)
#
# clf_nn=MLP(hidden_layer_sizes=5,activation="logistic")
# clf_nn.fit(train_feature,train_target)
# test_y_predict=clf_nn.predict(test_feature)

# MAPE=np.sum((abs(test_target-test_y_predict)/test_target*100))/len(test_target)
# print("MAPE:",MAPE)



#驗證模型好壞
#將資料切割為訓練資料與測試資料
#form sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)

#查看學習曲線
#from sklearn.learning_curve import learning_curve
#