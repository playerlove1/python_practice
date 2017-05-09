#測試多筆資料

import os
import numpy as np
from tools import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

#目前的目錄
root_dir=os.getcwd()

#所有資料的根目錄
data_dir="data1"+"\\"

#data1目錄下的所有VID
all_location_dir_list=os.listdir(data_dir)

train_dir="\\"+"train"+"\\"
test_dir="\\"+"test"+"\\"

#資料檔的名稱
data_name_xls="pems_output.xls"


print(all_location_dir_list)


#VIS的ID list
VDS_IDs=[]
MAPE=[]
MAPE_combine=[]
AIC=[]
AIC_combine=[]
BIC=[]
BIC_combine=[]
method=['SVR','MLP','GPR']

#對所有VDS 進行迴圈
for location in all_location_dir_list:
    VDS_IDs.append(location)
    train_data_dir=data_dir+location+train_dir
    test_data_dir=data_dir+location+test_dir
    train_data_dir_list=os.listdir(train_data_dir)
    test_data_dir_list=os.listdir(test_data_dir)
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
    
    
    #取出feature與target 並轉為np array 提供機器學習演算法運算
    train_data_combine_feature=(train_data_combine.iloc[:,0:8]).as_matrix(columns=None)
    train_data_combine_target=(train_data_combine.iloc[:,8]).as_matrix(columns=None)
    test_data_combine_feature=(test_data_combine.iloc[:,0:8]).as_matrix(columns=None)
    test_data_combine_target=(test_data_combine.iloc[:,8]).as_matrix(columns=None)
    
    
    
    
    
    # SVR
    
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
    
    draw_picture_compare(test_data_combine_feature,test_data_combine_target,predict,predict_combine,'SVR-'+str(location),str(location)+"_SVR.png")
    # if(location=='400868'):
        # print(test_data_target,predict)
    MAPE.append(cal_mape(test_data_target,predict))
    MAPE_combine.append(cal_mape(test_data_combine_target,predict_combine))
    
    AIC.append(cal_aic(test_data_target,predict,2))
    AIC_combine.append(cal_aic(test_data_combine_target,predict_combine,2))
    #BIC=log(e mle)+ d log(n)/n
    BIC.append(cal_bic(test_data_target,predict,2))
    BIC_combine.append(cal_bic(test_data_combine_target,predict_combine,2))
    
    # MLP
    clf_nn=MLP(hidden_layer_sizes=5,activation="logistic")
    
    predict_mlp=cal_mlp(clf_nn,train_data,test_data)
    predict_mlp_combine=cal_mlp_combine(clf_nn,train_data_combine,test_data_combine)
    
    draw_picture_compare(test_data_combine_feature,test_data_combine_target,predict_mlp,predict_mlp_combine,'MLPR-'+str(location),str(location)+"_MLP.png")
    
    MAPE.append(cal_mape(test_data_target,predict_mlp))
  
    MAPE_combine.append(cal_mape(test_data_combine_target,predict_mlp_combine))
    
    AIC.append(cal_aic(test_data_target,predict_mlp,6))
    AIC_combine.append(cal_aic(test_data_combine_target,predict_mlp_combine,6))

    BIC.append(cal_bic(test_data_target,predict_mlp,6))
    BIC_combine.append(cal_bic(test_data_combine_target,predict_mlp_combine,6))
    
    #GPR
    gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    + WhiteKernel(1e-1)
    clf_gpr=GPR(kernel=gp_kernel)
    predict_gpr=cal_gpr(clf_gpr,train_data,test_data)
    predict_gpr_combine=cal_gpr_combine(clf_gpr,train_data_combine,test_data_combine)
    
    draw_picture_compare(test_data_combine_feature,test_data_combine_target,predict_gpr,predict_gpr_combine,'GPR-'+str(location),str(location)+"_GPR.png")
    
    MAPE.append(cal_mape(test_data_target,predict_gpr))
    MAPE_combine.append(cal_mape(test_data_combine_target,predict_gpr_combine))
    
    AIC.append(cal_aic(test_data_target,predict_gpr,2))
    AIC_combine.append(cal_aic(test_data_combine_target,predict_gpr_combine,2))

    BIC.append(cal_bic(test_data_target,predict_gpr,2))
    BIC_combine.append(cal_bic(test_data_combine_target,predict_gpr_combine,2))
    
    
    

save_file(VDS_IDs,"MAPE",MAPE)
    
save_file(VDS_IDs,"MAPE_combine",MAPE_combine)

save_file(VDS_IDs,"AIC",AIC)

save_file(VDS_IDs,"AIC_combine",AIC_combine)

save_file(VDS_IDs,"BIC",BIC)

save_file(VDS_IDs,"BIC_combine",BIC_combine)


