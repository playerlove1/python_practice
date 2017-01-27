#小鬧鐘-定時開啟瀏覽器至youtube播放音樂

#引入必要模組  
import webbrowser
import time


#總共要休息的次數
total_breaks=3
#紀錄目前休息到第幾次
break_count=0
#程式要等待的時間(秒)   預設為2小時
time_wait_second=2*60*60


#程式啟動的顯示字串
print("This is program started on "+time.ctime())

while(break_count<total_breaks)
    #等待時間
    time.sleep(time_wait_second)
    #開啟瀏覽器與指定URL
    webbrowser.open("https://www.youtube.com/watch?v=wSBXfzgqHtE")
    break_count+=1
