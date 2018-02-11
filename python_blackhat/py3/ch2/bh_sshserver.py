import socket
import paramiko
import threading
import sys

# 使用Paramiko範例程式中的key
host_key = paramiko.RSAKey(filename='test_rsa.key')


# 定義伺服器的自訂類別
class Server(paramiko.ServerInterface):
    
    # 建構式
    def __init__(self):
        self.event = threading.Event()
    
    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
    # 檢查使用者的帳號密碼
    def check_auth_password(self, username, password):
        # 若帳號密碼皆為test
        if (username == 'test') and (password == 'test'):
            # 回傳授權成功
            return paramiko.AUTH_SUCCESSFUL
        # 回傳授權失敗
        return paramiko.AUTH_FAILED

# 使用方法的函數 (用print的方式顯示出用法)
def usage():
    print("-----Linux-----")
    print("Usage: ./bh_sshserver.py [ip] [port] ")
    print("Example: ./bh_sshserver.py 127.0.0.1 22 ")
    print("-----Windows-----")
    print("Usage: .\\bh_sshserver.py [ip] [port]")
    print("Example: .\\bh_sshserver.py 127.0.0.1 22")
    sys.exit(0)
        

def main():
    # 若由命令列讀入的參數長度不正確則顯示使用方法
    if len(sys.argv[1:]) != 2:
        usage()
    # 由命令列讀取相關參數
    server=sys.argv[1]
    ssh_port = int(sys.argv[2])
    
    # 建立Server 並開始監聽
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((server, ssh_port))
        sock.listen(100)
        print('[+] Listening for connection...')
        client, addr = sock.accept()
    except Exception as e:
        print('[-] Listen failed: ' + str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)

    print('[+] Got a connection!')
    
    try:
        # 開始使用key進行加密傳輸
        bhSession = paramiko.Transport(client)
        bhSession.add_server_key(host_key)
        server = Server()
        try:
            bhSession.start_server(server=server)
        except paramiko.SSHException as x:
            print('[-] SSH negotiation failed.')
        # 成功授權
        chan = bhSession.accept(20)
        print('[+] Authenticated!')
        print(chan.recv(1024).decode())
        # 傳送歡迎字串
        chan.send(b'Welcome to bh_ssh')
        
        while True:
            try:
                # 接收使用者輸入的command
                command = input('Enter command: ').strip('\n')
                
                # 當command不為exit的時候就傳送
                if command != 'exit':
                    chan.send((command+'\n').encode())
                    print(chan.recv(1024).decode() + '\n')
                # 否則傳送exit並關閉session
                else:
                    chan.send(b'exit')
                    print('exiting')
                    bhSession.close()
                    raise Exception('exit')
            except KeyboardInterrupt:
                bhSession.close()
                raise
    except Exception as e:
        print('[-] Caught exception: ' + str(e))
        try:
            bhSession.close()
        except:
            pass

    sys.exit(1)
main()