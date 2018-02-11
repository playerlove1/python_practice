#利用python 來取代NetCat
import sys
import socket
import getopt
import threading
import subprocess
import platform

# 定義一些全域變數
listen             = False
command            = False
upload             = False
execute            = ""
target             = ""
upload_destination = ""
port               = 0



# 執行命令與回傳結果
def run_command(command):

        # 裁掉換行符號
        command = command.rstrip()
        # 執行指令並取回輸出
        try:
                output = subprocess.check_output(command,stderr=subprocess.STDOUT, shell=True)
                if platform.system() == 'Windows':
                    # 將執行shell command回傳的bytes字串以cp950(windows cmd default encoding)編碼成字串 
                    output_encoding = 'cp950'
                    output = output.decode(output_encoding)
                    
        except:
                output = "Failed to execute command.\r\n"
        
        # 把輸出的字串傳回用戶端
        return output

# 處理進來的client連線
def client_handler(client_socket):
        global upload
        global execute
        global command

        # 檢查上傳
        if len(upload_destination):
                
                # 讀入所有bytes 並寫到指定位置
                file_buffer = ""
                
                # 一直讀到沒有資料可以讀為止
                while True:
                        data = client_socket.recv(1024)
                        
                        if not data:
                                break
                        else:
                                file_buffer += data
                                
                # 然後試著把這些資料儲存到檔案
                try:
                        file_descriptor = open(upload_destination,"wb")
                        file_descriptor.write(file_buffer)
                        file_descriptor.close()
                        
                        # 回應我們確實把資料存成檔案了
                        client_socket.send(b"Successfully saved file to %s\r\n" % upload_destination)
                except:
                        client_socket.send(b"Failed to save file to %s\r\n" % upload_destination)
                        

        # 檢查執行命令
        if len(execute):
                
                # 執行命令
                output = run_command(execute)
                client_socket.send(output.encode())
                client_socket.close()
        
        
        # 如果要求shell，就進入另一個迴圈
        if command:
                # 顯示一個簡單的提示
                client_socket.send(b"Connect to the command...  ")
                
                while True:
                        # 接著持續接收資料，直到收到LF(Enterr鍵)
                        cmd_buffer = ""
                        while "\n" not in cmd_buffer:
                                cmd_buffer += client_socket.recv(1024).decode()
                        if cmd_buffer == 'q\n':
                            sys.exit(0)
                        # 取得指令輸出
                        response = run_command(cmd_buffer)
                        # 顯示取得的指令
                        print(cmd_buffer)
                        # 顯示執行指令後的內容
                        print(response)
                        # 回傳
                        client_socket.send(response.encode())
            
# 針對進來的連線做處理
def server_loop():
        global target
        global port
        
        # 如果沒有定義目標，就監聽所有介面
        if not len(target):
                target = "0.0.0.0"
                
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((target,port))
        
        server.listen(5)        
        
        while True:
                client_socket, addr = server.accept()
                
                # 啟動一個Thread處理新的client
                client_thread = threading.Thread(target=client_handler,args=(client_socket,))
                client_thread.start()
                

# if we don't listen we are a client....make it so.
def client_sender(buffer):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
        try:
                # 連線到目標主機
                client.connect((target,port))
                
                # 如果偵測到有輸入資料在標準輸入 就傳送它
                # 若沒有就持續等待 直到使用者打入某些資料
                
                if len(buffer):
                        client.send(buffer.encode()) 
                while True:
                        
                        # 等待資料回傳
                        recv_len = 1
                        response = ""
                        
                        while recv_len:
                                data     = client.recv(4096)
                                recv_len = len(data)
                                response+= data.decode()
                                print(response)
                                
                                if recv_len < 4096:
                                        break

                        # 等待更多的輸入
                        buffer = input("<BHP:#> ")
                        if buffer == 'q':
                            sys.exit(0)
                        else:
                            buffer += "\n"                        
                        
                            # 傳送出去
                            client.send(buffer.encode())

                        
                
        except Exception as e:
                # just catch generic errors - you can do your homework to beef this up
                print("[*] Exception! Exiting.")
                print(type(e),str(e))
                # teardown the connection                  
                client.close()  
                        
                        
        
#使用方法的函數 (用print的方式顯示出用法)
def usage():
        print("Netcat Replacement")
        print("")
        print("usage: bhpnet.py -t target_host -p port")
        print("-l --listen                - listen on [host]:[port] for incoming connections")
        print("-e --execute=file_to_run   - execute the given file upon receiving a connection")
        print("-c --command               - initialize a command shell")
        print("-u --upload=destination    - upon receiving connection upload a file and write to [destination]")
        print("")
        print("")
        print("Examples: ")
        print("bhpnet.py -t 192.168.0.1 -p 5555 -l -c")
        print("bhpnet.py -t 192.168.0.1 -p 5555 -l -u=c:\\target.exe")
        print("bhpnet.py -t 192.168.0.1 -p 5555 -l -e=\"cat /etc/passwd\"")
        print("echo 'ABCDEFGHI' | ./bhpnet.py -t 192.168.11.12 -p 135")
        sys.exit(0)


def main():
        global listen
        global port
        global execute
        global command
        global upload_destination
        global target

            
        if not len(sys.argv[1:]):
                usage()
                
        # 讀取 commandline 的參數選項
        try:
                opts, args = getopt.getopt(sys.argv[1:],"hle:t:p:cu:",["help","listen","execute","target","port","command","upload"])
        except getopt.GetoptError as err:
                print(str(err))
                usage()
                
                
        for o,a in opts:
                if o in ("-h","--help"):
                        usage()
                elif o in ("-l","--listen"):
                        listen = True
                elif o in ("-e", "--execute"):
                        execute = a
                elif o in ("-c", "--commandshell"):
                        command = True
                elif o in ("-u", "--upload"):
                        upload_destination = a
                elif o in ("-t", "--target"):
                        target = a
                elif o in ("-p", "--port"):
                        port = int(a)
                else:
                        assert False,"Unhandled Option"
        

        # 我們要監聽還是只是從 stdin傳送資料
        if not listen and len(target) and port > 0:
                # 從命令列讀入buffer
                # 這會block，如果沒有透過stdin傳送資料的話
                # 要按CTRL+D(Linux)  CTRL+Z(Windows)
                buffer = sys.stdin.read()
                
                # 把資料送出去
                client_sender(buffer)   
        # 我們要監聽，同時可能根據上面的命令列選項
        # 上傳東西、執行指令，或是提供shell
        if listen:
                server_loop()

main()       