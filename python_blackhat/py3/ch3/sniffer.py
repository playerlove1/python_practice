import socket
import os

# 監聽的主機
host = "10.1.202.138"

# 建立raw socket綁定到公開介面
if os.name == "nt":
    socket_protocol = socket.IPPROTO_IP
else:
    socket_protocol = socket.IPPROTO_ICMP

sniffer = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket_protocol)

sniffer.bind((host, 0))

# 我們希望捕捉內容包含IP headers
sniffer.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

# 如果使用Windows則需要發送一個IOCTL
# 設定混雜模式
if os.name == "nt":
    sniffer.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)

# 讀取一個封包
print(sniffer.recvfrom(65565))


# 如果使用Windows，關閉混雜模式
if os.name == "nt":
    sniffer.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)