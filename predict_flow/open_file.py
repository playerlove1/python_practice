import os
import pandas as pd
import xlrd
import csv


#feature:time series(1,2,3,..) 看到哪個點   target:下個時間點的資料
#時序分析資料

#傳入參數
def format_data(windows_size,move_size,data_lists):
    
    all_df=[]
    
    for data_list in data_lists:
        
        new_data_frame_cols=windows_size
                                # //整數除法
        new_data_frame_rows=(len(data_list)-windows_size)//move_size +1
            
        all_list=[]
           
        row_list=[]
       
        for j in range(0,new_data_frame_rows-1):
            for k in range(1,new_data_frame_cols+1):
                row_list.append(int(data_list.iloc[j*move_size+k,:]))
                if (k%new_data_frame_cols==0):
                    all_list.append(row_list)
                   # print(row_list)
                    row_list=[]
                    
         
        
        #產生欄位名稱
        c=[]
        for j in range(1,new_data_frame_cols+1):
            c.append('data'+str(j))
        
        df_single=pd.DataFrame(all_list,columns=c)
        
        all_df.append(df_single)
    
    return all_df
    


#目前的目錄
root_dir=os.getcwd()

#放檔案的目錄路徑
data_dir="data"

#data目錄下各資料夾的list
data_dir_list=os.listdir(data_dir)
#資料的名稱
data_name="pems_output.csv"


#顯示目前的根目錄路徑
print(root_dir)

#顯示data內資料夾的名稱list
print(data_dir_list)

file_list=[]



#讀出資料
for data_folder in data_dir_list:
    #print(data_folder+"\\"+data_name)
    file_list.append(pd.read_csv(data_dir+"\\"+data_folder+"\\"+data_name))

i=0
#切割每個資料集的特定欄位
for file_dataframe in file_list:
    num=len(file_dataframe.columns)
    #print(file_dataframe.iloc[:,0:1])
    file_list[i]=file_dataframe.iloc[:,num-3:num-2]#pd.concat([file_dataframe.iloc[:,0:1],file_dataframe.iloc[:,num-3:num-2]],axis=1)
    i=i+1
    
    
#print(file_list) 
all_df=format_data(5,1,file_list)

print(all_df)
#






