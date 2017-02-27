#測試自訂函數tool
import os
import numpy as np
from sklearn.svm import SVR
from tools import *
from sklearn import preprocessing

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

c=[]
for j in range(1,10):
    c.append('data'+str(j))

train_data=pd.DataFrame(preprocessing.scale(train_data),columns=c)

#取出feature與target 並轉為np array 提供機器學習演算法運算
train_feature=(train_data.iloc[:,0:8]).as_matrix(columns=None)
train_target=(train_data.iloc[:,8]).as_matrix(columns=None)


#測試learning_curve
draw_learning_curve(SVR(kernel="rbf",C=1,gamma=0.1), train_feature, train_target)


#測試validation_curve
p_range = np.geomspace(1,100,num=10)
draw_validation_curve(SVR(gamma=0.0001), train_feature, train_target, 'C', p_range)


#測試grid_search
parameter_candidates = {'kernel':('linear', 'rbf'), 'C':[1,10,100,1000], 'gamma':[0.1,0.01,0.001,0.0001]}
show_grid_search(SVR(), parameter_candidates, train_feature, train_target)
