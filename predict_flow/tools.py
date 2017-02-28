#所有相關需要用的函式集合成的檔案

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.learning_curve import  validation_curve
from sklearn.learning_curve import  learning_curve
from sklearn.grid_search import GridSearchCV
import numpy as np
import math

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
#根據 演算法 特徵與目標(測試資料)  做出圖   
def draw_picture_with_clf(clf,test_feature,test_target):
    y_predict=clf.predict(test_feature)
    #將結果化成圖
    x=np.arange(len(test_feature))
    draw_picture(x,test_target,y_predict)
    return y_predict

#畫出學習曲線(learning_curve)  傳入參數  演算法,特徵,目標
def draw_learning_curve(clf, train_feature, train_target):
   train_sizes, train_loss, test_loss= learning_curve(clf, train_feature, train_target, cv=10, scoring='mean_squared_error',
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

#畫出該參數不同range的準確度  傳入參數  演算法,特徵,目標,參數名稱,參數範圍
def draw_validation_curve(clf, train_feature, train_target, param_name, param_range):


    train_loss, test_loss = validation_curve(clf, train_feature, train_target, param_name=param_name, param_range=param_range, cv=10,
        scoring='mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color="r",
                 label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g",
                 label="Cross-validation")

    plt.xlabel(param_name)
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
#顯示網格搜尋後最佳的參數
def show_grid_search(clf, parameter_candidates, train_feature, train_target):
    # parameter_candidates = {'kernel':('linear', 'rbf'), 'C':[1,10,100,1000], 'gamma':[0.1,0.01,0.001,0.0001]}
    
    # Create a classifier with the parameter candidates
    clf_cv = GridSearchCV(estimator=clf, param_grid=parameter_candidates,cv=5)
    clf_cv.fit(train_feature, train_target)
    print('Best score for training data:', clf_cv.best_score_)
    print('Best `C`:',clf_cv.best_estimator_.C)
    print('Best kernel:',clf_cv.best_estimator_.kernel)
    print('Best `gamma`:',clf_cv.best_estimator_.gamma)
    
 #MAPE sigma( (實際值-預測值)/ 實際值  *100 )/資料筆數
def cal_mape(y,pred_y):
   MAPE=np.sum((abs(y-pred_y)/y*100))/len(y)
   return MAPE
#AIC = log(e mle)+ 2d/n
#n:資料筆數 d:參數個數  e mle=sigma((實際值-預測值)^2)/資料筆數
def cal_aic(y,pred_y,d):
    e_mle=np.sum((abs(y-pred_y)**2))/len(y)
    aic=math.log(e_mle)+d/len(y)
    return aic
#BIC=log(e mle)+ d log(n)/n
#n:資料筆數 d:參數個數  e mle=sigma((實際值-預測值)^2)/資料筆數
def cal_bic(y,pred_y,d):
    e_mle=np.sum((abs(y-pred_y)**2))/len(y)
    bic=math.log(e_mle)+(d*math.log(len(y))/len(y))
    return bic 