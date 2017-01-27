#本範例為將 prank資料夾中的檔案檔名做修改  若有包含數字則將數字去除
#prank為原始資料  prank-test為利用原始資料複製出來的同個 資料夾以便進行操作看結果

import os  

import shutil

#定義函式 
def rename_files():

    #執行時的資料夾
    file_path_exc="prank-test"
    
    #Step1 - get file nmaes from a folder

    #利用相對路徑讀取該資料夾內的所有檔案  在前面加上r表示讓python知道直接讀取該路徑即可不須針對字串做處理
    file_list=os.listdir(r"prank")
    #print(file_list)
    saved_path=os.getcwd()
    print("Current Working Directory is "+saved_path)

    #檢查執行資料夾是否存在 若存在則先移除再複製
    if not os.path.isdir(file_path_exc):
        #複製原始資料夾
        shutil.copytree('prank', 'prank-test')
    else:
         #移除再複製原始資料夾
        shutil.rmtree(file_path_exc)
        shutil.copytree('prank', 'prank-test')
    #切換至該目錄並進行修改
    os.chdir(r"prank-test")
    #Step2 - for each file,rename filename
    for file_name in file_list:
        #透過translate函式 將0-9的數字移除
        print("Old name -",file_name)
        print("New name -",file_name.translate("0123456789"))
        os.rename(file_name,file_name.translate("0123456789"))
    #os.chdir(saved_path)
    
rename_files()
