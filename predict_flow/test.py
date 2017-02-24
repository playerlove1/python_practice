#測試單一檔案並測試預測
import os
import pandas as pd
import xlrd
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np


#將時序資料依據windows size 與move size 移動取出feature與target
def format_data(windows_size,move_size,data_list):

    new_data_frame_cols=windows_size
                            # //整數除法
    new_data_frame_rows=(len(data_list)-windows_size)//move_size +1
        
    all_list=[]
       
    row_list=[]
   
    for j in range(0,new_data_frame_rows):
        for k in range(1,new_data_frame_cols+1):
            row_list.append(int(data_list.iloc[j*move_size+k-1,:]))
            if (k%new_data_frame_cols==0):
                all_list.append(row_list)
                #print(row_list)
                row_list=[]
                
     
    
    #產生欄位名稱
    c=[]
    for j in range(1,new_data_frame_cols+1):
        c.append('data'+str(j))
    
    
    
    df_single=pd.DataFrame(all_list,columns=c)
    return df_single

def draw_picture(feature,target,y_predict):
    plt.scatter(feature, target, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(feature, y_predict, color='navy',lw=2,label='predict')
    plt.xlabel('time')
    plt.ylabel('veh/5 Minutes')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

       

        





#目前的目錄
root_dir=os.getcwd()

#放檔案的目錄路徑
data_dir="data"

#data目錄下各資料夾的list
data_dir_list=os.listdir(data_dir)
#資料的名稱
data_name="pems_output.csv"


file_csv=pd.read_csv(data_dir+"\\"+"400300"+"\\"+data_name)
# file_csv=pd.read_csv(data_dir+"\\"+"716091"+"\\"+data_name)



#取出流量資料
num=len(file_csv.columns)
file_target=file_csv.iloc[:,num-3:num-2]

# print(file_target)

df=format_data(5,1,file_target)  

feature=(df.iloc[:,0:4]).as_matrix(columns=None)
target=(df.iloc[:,4]).as_matrix(columns=None)

print(feature)
print(len(target))

clf = SVR(kernel="rbf",C=100)


clf.fit(feature,target)
y_predict=clf.predict(feature)
x=np.arange(len(target))
draw_picture(x,target,y_predict)


# data=[233,264,241,274,289,287,297]

# df=pd.DataFrame(data,columns={'time'})

# print(df)
# format_data(5,1,data)