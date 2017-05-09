#單一個VDS資料的測試
import os
import numpy as np
from tools import *
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ExpSineSquared
from sklearn.grid_search import GridSearchCV

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

# train_data_dir="data1"+"\\"+"1000810"+"\\"+"train"+"\\"
# test_data_dir="data1"+"\\"+"1000810"+"\\"+"test"+"\\"

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



# gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    # + WhiteKernel(1e-1)
gp_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    
clf_gpr=GPR(kernel=gp_kernel,alpha=0.001)
# clf_gpr=GPR()

predict_gpr=cal_gpr(clf_gpr,train_data,test_data)
predict_gpr_combine=cal_gpr_combine(clf_gpr,train_data_combine,test_data_combine)



draw_picture_compare(test_data_combine_feature,test_data_combine_target,predict_gpr,predict_gpr_combine,'GaussianProcessRegressor','test.png')