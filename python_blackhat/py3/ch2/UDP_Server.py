#UDP Server example
import socket
import sys

# 執行Server的迴圈
def server_loop(bind_ip, bind_port):
    # 建立一個UDP的socket物件
    '''
    AF_INET 代表我們將採用標準IPv4的位址或是主機名稱
    SOCK_DGRAM 代表這是一個UDP client
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 建立一個綁定socket的 ip與port
    server_address = (bind_ip, bind_port)
    
    # 綁定要讓伺服器監聽的ip與port
    sock.bind(server_address)
    
    # 顯示目前在哪個ip與port進行監聽
    print('starting up on {} port {}'.format(*server_address))

    while True:
        # 顯示等待連線訊息的字串
        print('\nwaiting to receive message')
        
        # 接收連線訊息與傳送的資料
        data, address = sock.recvfrom(4096)
        # 顯示由何處接收多少長度的資料
        print('received {} bytes from {}'.format( len(data), address))
        # 顯示接收到資料的內容\
        print(data.decode())
        
        # 如果有接收到資料
        if data:
            # 回傳剛剛所接收到的資料
            sent = sock.sendto(data, address)
            # 顯示回傳的資料
            print('sent {} bytes back to {}'.format( sent, address))

            
#使用方法的函數 (用print的方式顯示出用法)
def usage():
    print("-----Linux-----")
    print("Usage: ./UDP_Server.py [ip] [port]")
    print("Example: ./UDP_Server.py 127.0.0.1 9999")
    print("-----Windows-----")
    print("Usage: .\\UDP_Server.py [ip] [port]")
    print("Example: .\\UDP_Server.py 127.0.0.1 9999")
    sys.exit(0)

    
def main():
    # 沒有華麗的命令列解釋
    if len(sys.argv[1:]) != 2:
        usage()
    #由命令列讀取相關參數 
    bind_ip = sys.argv[1]
    bind_port = int(sys.argv[2])
    # 開始執行Server的迴圈
    server_loop(bind_ip, bind_port)

main()