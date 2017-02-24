#測試參數看結果
from sklearn.learning_curve import  learning_curve
import os
import numpy as np
from tools import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP
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



c=[]
for j in range(1,10):
    c.append('data'+str(j))

train_data=pd.DataFrame(preprocessing.scale(train_data),columns=c)

print(train_data)
#取出feature與target 並轉為np array 提供機器學習演算法運算
train_feature=(train_data.iloc[:,0:8]).as_matrix(columns=None)
train_target=(train_data.iloc[:,8]).as_matrix(columns=None)


train_sizes, train_loss, test_loss= learning_curve(
        SVR(kernel="rbf",C=1,gamma=0.1), train_feature, train_target, cv=10, scoring='mean_squared_error',
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])


train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
             label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
             label="Cross-validation")


plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

print(test_loss_mean)


#high variance
# smaller feature , more data , increasing

