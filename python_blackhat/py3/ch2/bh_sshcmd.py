import threading
import paramiko
import subprocess

def ssh_command(ip, user, passwd, command):
    
    # 建立ssh的Client端
    client = paramiko.SSHClient()
    
    #client.load_host_keys('/home/justin/.ssh/known_hosts')
    # 使用傳統的使用者名稱與密碼的方式進行認證
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # 發起ssh連線
    client.connect(ip, username=user, password=passwd)
    
    # 開啟session
    ssh_session = client.get_transport().open_session()
    
    # 當連線成功時就執行命令
    if ssh_session.active:
        ssh_session.exec_command(command)
        
        print(ssh_session.recv(1024).decode())
    
    return

ssh_command('192.168.137.2', 'test', 'test', 'id')