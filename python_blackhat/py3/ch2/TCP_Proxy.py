import sys
import socket
import threading


'''
# 直接從下列網址註解中取得的16進為美觀傾印的函式
# http://code.activestate.com/recipes/142812-hex-dumper/
def hexdump(src, length=16):
    result = []
    digits = 4 if isinstance(src, unicode) else 2

    for i in xrange(0, len(src), length):
       s = src[i:i+length]
       hexa = b' '.join(["%0*X" % (digits, ord(x))  for x in s])
       text = b''.join([x if 0x20 <= ord(x) < 0x7F else b'.'  for x in s])
       result.append( b"%04X   %-*s   %s" % (i, length*(digits + 1), hexa, text) )

    print(b'\n'.join(result))
'''
def hexdump( src, length=16, sep='.' ):
    '''
    @brief Return {src} in hex dump.
    @param[in] length   {Int} Nb Bytes by row.
    @param[in] sep      {Char} For the text part, {sep} will be used for non ASCII char.
    @return {Str} The hexdump
    @note Full support for python2 and python3 !
    '''
    result = [];

    # Python3 support
    try:
        xrange(0,1);
    except NameError:
        xrange = range;

    for i in xrange(0, len(src), length):
        subSrc = src[i:i+length]
        hexa = ''
        isMiddle = False
        for h in xrange(0,len(subSrc)):
            if h == length/2:
                hexa += ' '
            h = subSrc[h]
            if not isinstance(h, int):
                h = ord(h)
            h = hex(h).replace('0x','')
            if len(h) == 1:
                h = '0'+h;
            hexa += h+' '
        hexa = hexa.strip(' ')
        text = ''
        for c in subSrc:
            if not isinstance(c, int):
                c = ord(c)
            if 0x20 <= c < 0x7F:
                text += chr(c)
            else:
                text += sep
        result.append(('%08X:  %-'+str(length*(2+1)+1)+'s  |%s|') % (i, hexa, text))
    print('\n'.join(result))


def receive_from(connection):
    
    buffer = ""

    # 我們設定2秒timeout
    # 這可能要根據測試對象調整
    connection.settimeout(10)
    
    try:
        # 持續讀取buffer直到沒有資料
        # 或是timeout
        while True:

            data = connection.recv(4096)

            buffer += data.decode()

            if not data:
                break
                    
            

    except:
        pass
    
    return buffer.encode()

# 修改要傳送到遠端的要求
def request_handler(buffer):
    # 進行修改動作
    return buffer

# 修改回傳本機的資料
def response_handler(buffer):
    # 進行修改的動作
    return buffer


def proxy_handler(client_socket, remote_host, remote_port, receive_first):

    # 連接遠端主機
    remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    remote_socket.connect((remote_host,remote_port))

    # 如果有需要先從遠端主機接收資料
    if receive_first:
        print('receive first')
        remote_buffer = receive_from(remote_socket)
        hexdump(remote_buffer)
        
        # 送到處理回應的函式
        remote_buffer = response_handler(remote_buffer)
                
        # 如果有資料要回傳，就回傳
        if len(remote_buffer):
            print("[<==] Sending %d bytes to localhost." % len(remote_buffer))
            client_socket.send(remote_buffer)
                        
    # 進入迴圈並從localhost讀取 傳送到遠端
    # 反覆不斷
    while True:
        
        # 由local host 讀取
        local_buffer = receive_from(client_socket)


        if len(local_buffer):   
            
            print("[==>] Received %d bytes from localhost." % len(local_buffer))
            hexdump(local_buffer)
            
            # 送出處理請求的函式
            local_buffer = request_handler(local_buffer)
            
            # 把資料送到遠端
            remote_socket.send(local_buffer)
            print("[==>] Sent to remote.")
        
        
        # 接收回應
        remote_buffer = receive_from(remote_socket)

        if len(remote_buffer):
            
            print("[<==] Received %d bytes from remote." % len(remote_buffer))
            hexdump(remote_buffer)
            
            # 送到處理回應的函式
            remote_buffer = response_handler(remote_buffer)
        
            # 把回應送localhost的socket
            client_socket.send(remote_buffer)
            
            print("[<==] Sent to localhost.")
        
        # 如果兩邊都沒有後續資料，就關掉連線
        if not len(local_buffer) or not len(remote_buffer):
            client_socket.close()
            remote_socket.close()
            print("[*] No more data. Closing connections.")
            break
        
def server_loop(local_host,local_port,remote_host,remote_port,receive_first):

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind((local_host,local_port))
    except:
        print("[!!] Failed to listen on %s:%d" % (local_host,local_port))
        print("[!!] Check for other listening sockets or correct permissions.")
        sys.exit(0)
        
    print("[*] Listening on %s:%d" % (local_host,local_port))

    server.listen(5)        

    while True:
        client_socket, addr = server.accept()
               
        # 顯示本機連線資訊
        print("[==>] Received incoming connection from %s:%d" % (addr[0],addr[1]))

        # 啟用一個thread與遠端溝通
        proxy_thread = threading.Thread(target=proxy_handler,args=(client_socket,remote_host,remote_port,receive_first))
        proxy_thread.start()

        
def usage():
    print("Usage: .\TCP_Proxy.py [localhost] [localport] [remotehost] [remoteport] [receive_first]")
    print("Example: .\TCP_Proxy.py 127.0.0.1 8888 10.12.132.1 9999 True")
    sys.exit(0)

def main():
        
    # 若由命令列讀入的參數長度不正確則顯示使用方法
    if len(sys.argv[1:]) != 5:
       usage()
    
    # 設定本機的監聽資訊
    local_host  = sys.argv[1]
    local_port  = int(sys.argv[2])
    
    # 設定遠端資訊
    remote_host = sys.argv[3]
    remote_port = int(sys.argv[4])
    
    # 告訴proxy先連上遠端主機、接收資料
    # 再傳送回來
    receive_first = sys.argv[5]
    
    if "True" in receive_first:
        receive_first = True
    else:
        receive_first = False
        
    
    # 建立我們監聽用的socket
    server_loop(local_host,local_port,remote_host,remote_port,receive_first)
        
main() 