#TCP Server example
'''
本範例為簡易的TCP Server
當接收到連線後會開啟一個執行序為該Client服務
當該執行序收到Client傳回的'q'訊息 則會離開while迴圈 結束處理程序
'''
import socket
import threading
import sys


#處理client的thread
def handle_client(client_socket):
    while True:
        # 取得Client傳入的訊息
        request = client_socket.recv(1024)
        
        # 若沒有訊息則結束
        if not request:
            break
        # 若收到'q' 則關閉此連線
        elif request.decode() == 'q' :
            # 印出關閉連線的訊息
            print('close this socket!')
            # 關閉連線
            client_socket.close()
            # 離開while True的無窮迴圈
            sys.exit(0)
            
        # 若是收到其他訊息
        else:
            # 印出關閉連線的訊息
            print("[*] Reveived: %s" % request.decode())
            # 回傳訊息給Client
            client_socket.send(b"got your message!\r\n")

# 執行Server的迴圈
def server_loop(bind_ip, bind_port):
    # 建立socket物件
    '''
    AF_INET 代表我們將採用標準IPv4的位址或是主機名稱
    SOCK_STREAM 代表這是一個TCP client
    '''
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 綁定要讓伺服器監聽的ip與port
    server.bind((bind_ip, bind_port))
    # 將排隊的上限數設為5
    server.listen(5)
    # 顯示目前在哪個ip與port進行監聽
    print("[*] Listening on %s:%d" % (bind_ip, bind_port))
    
    # 持續等待Client的連線
    # 當有Client連線後 啟動client handler
    while True:
        # 將client的socket接收並儲存至client的變數中 並將連線資訊存到addr變數
        client, addr = server.accept()
        
        print("[*] Accepted connection from: %s:%d" % (addr[0], addr[1]))
        
        # 啟動我們的client thread 處理傳來的資料
        client_handler = threading.Thread(target=handle_client,args=(client,))
        client_handler.start()

# 使用方法的函數 (用print的方式顯示出用法)
def usage():
    print("-----Linux-----")
    print("Usage: ./TCP_Server.py [ip] [port]")
    print("Example: ./TCP_Server.py 127.0.0.1 9999")
    print("-----Windows-----")
    print("Usage: .\\TCP_Server.py [ip] [port]")
    print("Example: .\\TCP_Server.py 127.0.0.1 9999")
    sys.exit(0)

def main():
    # 若由命令列讀入的參數長度不正確則顯示使用方法
    if len(sys.argv[1:]) != 2:
        usage()
    # 由命令列讀取相關參數 
    bind_ip = sys.argv[1]
    bind_port = int(sys.argv[2])
    # 開始執行Server的迴圈
    server_loop(bind_ip, bind_port)

main()