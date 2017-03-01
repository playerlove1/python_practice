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
train_data_dir="data"+"\\"+"400300"+"\\"+"train"+"\\"
test_data_dir="data"+"\\"+"400300"+"\\"+"test"+"\\"

#data目錄下各資料夾的list
train_data_dir_list=os.listdir(train_data_dir)
test_data_dir_list=os.listdir(test_data_dir)

#資料檔的名稱
data_name_xls="pems_output.xls"


#train資料
train_file_dfs=[]
#test資料
test_file_dfs=[]

#將train 的資料加入
for dir in train_data_dir_list:
    file_excel=get_time_flow(read_xls_file(train_data_dir+dir+"\\"+data_name_xls,'Report Data'))
    train_file_dfs.append(file_excel.iloc[:,1])


#將test的資料加入
for dir in test_data_dir_list:
    file_excel=get_time_flow(read_xls_file(test_data_dir+dir+"\\"+data_name_xls,'Report Data'))
    test_file_dfs.append(file_excel.iloc[:,1])
    
    
#格式化資料  

#原始資料轉為sliding windows的方式  
train_data=format_data(5,1,train_file_dfs[4])
test_data=format_data(5,1,test_file_dfs[0])



#取出feature與target 並轉為np array 提供機器學習演算法運算
train_data_feature=(train_data.iloc[:,0:4]).as_matrix(columns=None)
train_data_target=(train_data.iloc[:,4]).as_matrix(columns=None)
test_data_feature=(test_data.iloc[:,0:4]).as_matrix(columns=None)
test_data_target=(test_data.iloc[:,4]).as_matrix(columns=None)


#原始資料轉為sliding windows的方式 並加上歷史資料
train_data_combine=format_data_combine(5,1,train_file_dfs)
test_dfs=train_file_dfs[1:]+test_file_dfs
test_data_combine=format_data_combine(5,1,test_dfs)

# train_data_combine,test_data_combine=scale_data(train_data_combine,test_data_combine)

#取出feature與target 並轉為np array 提供機器學習演算法運算
train_data_combine_feature=(train_data_combine.iloc[:,0:8]).as_matrix(columns=None)
train_data_combine_target=(train_data_combine.iloc[:,8]).as_matrix(columns=None)
test_data_combine_feature=(test_data_combine.iloc[:,0:8]).as_matrix(columns=None)
test_data_combine_target=(test_data_combine.iloc[:,8]).as_matrix(columns=None)




#呼叫SVR演算法並指定參數
clf = SVR(kernel="rbf",C=100,gamma=0.0001)
#針對演算法 使用訓練資料fit (即訓練的意思)
clf.fit(train_data_feature,train_data_target)
# predict=draw_picture_with_clf(clf,test_data_feature,test_data_target)
predict=get_y_predict(clf,test_data_feature)


clf_combine= SVR(kernel="rbf",C=100,gamma=0.0001)
#針對演算法 使用訓練資料fit (即訓練的意思)
clf_combine.fit(train_data_combine_feature,train_data_combine_target)
# predict_combine=draw_picture_with_clf(clf_combine,test_data_combine_feature,test_data_combine_target)
predict_combine=get_y_predict(clf_combine,test_data_combine_feature)



draw_picture_compare(test_data_combine_feature,test_data_combine_target,predict,predict_combine)

#衡量結果之公式 
#MAPE sigma( (實際值-預測值)/ 實際值  *100 )/資料筆數
print("MAPE:",cal_mape(test_data_target,predict))
print("MAPE_combine:",cal_mape(test_data_combine_target,predict_combine))

# print("score:",clf.score(test_data_feature, test_data_target))

#AIC = log(e mle)+ 2d/n
#svr  c gamma 2個參數
print("aic:",cal_aic(test_data_target,predict,2))
print("aic_combine:",cal_aic(test_data_combine_target,predict_combine,2))
#BIC=log(e mle)+ d log(n)/n
print("bic:",cal_bic(test_data_target,predict,2))
print("bic_combine:",cal_bic(test_data_combine_target,predict_combine,2))


#缺其他演算法

#演算法
#NN (類神經網路)
clf_nn=MLP(hidden_layer_sizes=5,activation="logistic")
draw_mlp(clf_nn,train_data_combine,test_data_combine)
# clf_nn.fit(train_data_combine_feature,train_data_combine_target)

# MAPE=np.sum((abs(test_target-test_y_predict)/test_target*100))/len(test_target)
# print("MAPE:",MAPE)


# draw_picture_with_clf(clf_nn,test_data_combine_feature,test_data_combine_target)



#驗證模型好壞
#將資料切割為訓練資料與測試資料
#form sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)

#查看學習曲線
#from sklearn.learning_curve import learning_curve
#