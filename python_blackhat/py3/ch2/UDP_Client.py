#UDP Client example
import socket
import sys


def client_sender(target_host, target_port):

    # 建立一個UDP的socket物件
    '''
    AF_INET 代表我們將採用標準IPv4的位址或是主機名稱
    SOCK_DGRAM 代表這是一個UDP client
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_address = (target_host, target_port)
    message = b'This is the message.  It will be repeated.'

    try:
        # 傳送資料
        print('sending {!r}'.format(message.decode()))
        sent = sock.sendto(message, server_address)

        # 接收Server回應
        print('waiting to receive')
        data, server = sock.recvfrom(4096)
        print('received {!r}'.format(data.decode()))

    finally:
        print('closing socket')
        sock.close()
    
    
#使用方法的函數 (用print的方式顯示出用法)
def usage():
    print("-----Linux-----")
    print("Usage: ./UDP_Client.py [target_host] [target_port]")
    print("Example: ./UDP_Client.py 127.0.0.1 9999 True")
    print("-----Windows-----")
    print("Usage: .\\UDP_Client.py [target_host] [target_port]")
    print("Example: .\\UDP_Client.py 127.0.0.1 9999 True")
    sys.exit(0)
    
def main():
    # 沒有華麗的命令列解釋
    if len(sys.argv[1:]) != 2:
        usage()
    #由命令列讀取相關參數 
    target_host = sys.argv[1]
    target_port = int(sys.argv[2])
    
    client_sender(target_host, target_port)

main()