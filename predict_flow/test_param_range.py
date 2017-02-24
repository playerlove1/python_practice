#測試參數看結果

import os
import numpy as np
from tools import *
from sklearn.svm import SVR
from sklearn.learning_curve import  validation_curve
import matplotlib.pyplot as plt
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



# c=[]
# for j in range(1,10):
    # c.append('data'+str(j))

# train_data=pd.DataFrame(preprocessing.scale(train_data),columns=c)

print(train_data)
#取出feature與target 並轉為np array 提供機器學習演算法運算
train_feature=(train_data.iloc[:,0:8]).as_matrix(columns=None)
train_target=(train_data.iloc[:,8]).as_matrix(columns=None)



#帶不同的gamma畫圖看結果
# param_range = np.logspace(-6, -5,10 )

# train_loss, test_loss = validation_curve(
        # SVR(C=100), train_feature, train_target, param_name='gamma', param_range=param_range, cv=10,
        # scoring='mean_squared_error')
param_range = np.geomspace(1,100,num=10)

train_loss, test_loss = validation_curve(
        SVR(gamma=0.0001), train_feature, train_target, param_name='C', param_range=param_range, cv=10,
        scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

# plt.xlabel("gamma")
plt.xlabel("C")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

print(param_range)

#high variance
# smaller feature , more data , increasing

