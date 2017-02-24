#所有相關需要用的函式集合成的檔案

import pandas as pd
import matplotlib.pyplot as plt


#讀檔的函式 
def read_xls_file(filepath,sheetname):
    file_xls=pd.read_excel(filepath,'Report Data')
    return file_xls

#取出關鍵的時間與流量的欄位
def get_time_flow(file_content):
    #所有欄位的長度(即有幾行之意思)
    cols=len(file_content.columns)
    #時間在index 0  流量在col長度-3的位置
    new_content=file_content.iloc[:,[0,cols-3]]
    return new_content
 
#feature:time series(1,2,3,..) 看到哪個點   target:下個時間點的資料
#時序分析資料 利用滑動視窗法建立feature與target
#格式化資料 參數(滑動視窗的大小,滑動的步數,資料list)
def format_data(windows_size,move_size,data_list):
    

    new_data_frame_cols=windows_size
                                # //整數除法
    new_data_frame_rows=(len(data_list)-windows_size)//move_size +1
            
    all_list=[]
           
    row_list=[]
    
    for j in range(0,new_data_frame_rows-1):
        for k in range(0,new_data_frame_cols):
            row_list.append((data_list.iloc[j*move_size+k]))
            if (k%new_data_frame_cols==windows_size-move_size):
                all_list.append(row_list)
                # print(row_list)
                row_list=[]
                    
         
        
    #產生欄位名稱
    c=[]
    for j in range(1,new_data_frame_cols+1):
        c.append('data'+str(j))
        
    df_single=pd.DataFrame(all_list,columns=c)
        

    return df_single
def format_data_combine(windows_size,move_size,data_lists):    
    data_list=data_lists[len(data_lists)-1]
   
    new_data_frame_cols=windows_size
                                # //整數除法
    new_data_frame_rows=(len(data_list)-windows_size)//move_size +1
            
    all_list=[]
           
    row_list=[]
    

    row_left_list=[]
    
    for j in range(0,new_data_frame_rows):
        for k in range(0,4):
            row_left_list.append(data_lists[k].iloc[j+4])
            if(k%4==3):
                all_list.append(row_left_list)
                row_left_list=[]

    for j in range(0,new_data_frame_rows):
        for k in range(0,new_data_frame_cols):
            row_list.append((data_list.iloc[j*move_size+k]))
            if (k%new_data_frame_cols==windows_size-move_size):
                old_list=all_list[j]
                all_list[j]=old_list+row_list
                # print(row_list)
                row_list=[]
                    
         
        
    #產生欄位名稱
    c=[]
    for j in range(1,10):
        c.append('data'+str(j))
        
    df_single=pd.DataFrame(all_list,columns=c)
   
    
    return df_single


#根據 原始資料的x,y與預測資料  做出圖
def draw_picture(feature,target,y_predict):
    plt.scatter(feature, target, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(feature, y_predict, color='navy',lw=2,label='predict')
    plt.xlabel('time')
    plt.ylabel('veh/5 Minutes')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    
    