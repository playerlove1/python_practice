#TCP Client example
'''
本範例為簡易的TCP Client
若once參數為True則表示只傳送一次
若為False則會持續等待使用者輸入訊息直到按下'q'送出
'''
import socket
import sys

def client_sender(target_host, target_port, once):
    # 建立socket物件
    '''
    AF_INET 代表我們將採用標準IPv4的位址或是主機名稱
    SOCK_STREAM 代表這是一個TCP client
    '''
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 讓client連線
    client.connect((target_host, target_port))
    
    while True:
        # 讀取使用者輸入的資料
        data = input("please input data:")
        # 將資料進行編碼並傳送
        client.send(data.encode())
        
        # 如果資料為q 則離開
        if data == 'q':
            sys.exit(0)
            
        # 接收Server回傳的資料
        response = client.recv(4096)
        
        # 顯示server 回傳的資料
        print(response.decode())
        
        if once:
            # 傳送給Server中止Socket的訊息
            client.send(b"q")
            sys.exit(0)
            
# 使用方法的函數 (用print的方式顯示出用法)
def usage():
    print("-----Linux-----")
    print("Usage: ./TCP_Client.py [target_host] [target_port] [once]")
    print("Example: ./TCP_Client.py 127.0.0.1 9999 True")
    print("-----Windows-----")
    print("Usage: .\\TCP_Client.py [target_host] [target_port] [once]")
    print("Example: .\\TCP_Client.py 127.0.0.1 9999 True")
    sys.exit(0)
    
def main():
    # 若由命令列讀入的參數長度不正確則顯示使用方法
    if len(sys.argv[1:]) != 3:
        usage()
    # 由命令列讀取相關參數 
    target_host = sys.argv[1]
    target_port = int(sys.argv[2])
    once = sys.argv[3]
    
    if "True" in once:
        once = True
    else:
        once = False
    client_sender(target_host, target_port, once)

main()