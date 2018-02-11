import socket
import os
import struct
import threading
import time

from netaddr import IPNetwork,IPAddress
from ctypes import *

# 要監聽的主機
host   = "10.1.202.138"

# 要掃描的子網路
subnet = "10.1.202.0/24"

# 我們要在ICMP回應裡檢查的Magic string
magic_message = b"PYTHONRULES!"

# 這裡負責送出UDP datagrams
def udp_sender(subnet,magic_message):

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for ip in IPNetwork(subnet):
        try:
            # print("正在傳送%s"%ip)
            sender.sendto(magic_message,("%s" % ip,65212))
        except:
            pass
        
# 我們的IP header
class IP(Structure):
    
    _fields_ = [
        ("ihl",           c_ubyte, 4),
        ("version",       c_ubyte, 4),
        ("tos",           c_ubyte),
        ("len",           c_ushort),
        ("id",            c_ushort),
        ("offset",        c_ushort),
        ("ttl",           c_ubyte),
        ("protocol_num",  c_ubyte),
        ("sum",           c_ushort),
        ("src",           c_ulong),
        ("dst",           c_ulong)
    ]
    
    def __new__(self, socket_buffer=None):
            return self.from_buffer_copy(socket_buffer)    
        
    def __init__(self, socket_buffer=None):

        # 把協定常數對應到名稱
        self.protocol_map = {1:"ICMP", 6:"TCP", 17:"UDP"}
        
        # 人類可讀的IP位址
        self.src_address = socket.inet_ntoa(struct.pack("<L",self.src))
        self.dst_address = socket.inet_ntoa(struct.pack("<L",self.dst))
    
        # 人類可讀的協定
        try:
            self.protocol = self.protocol_map[self.protocol_num]
        except:
            self.protocol = str(self.protocol_num)
            


class ICMP(Structure):
    
    _fields_ = [
        ("type",         c_ubyte),
        ("code",         c_ubyte),
        ("checksum",     c_ushort),
        ("unused",       c_ushort),
        ("next_hop_mtu", c_ushort)
        ]
    
    def __new__(self, socket_buffer):
        return self.from_buffer_copy(socket_buffer)    

    def __init__(self, socket_buffer):
        pass

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


# 開始發送封包
t = threading.Thread(target=udp_sender,args=(subnet,magic_message))
t.start()        

try:
    while True:
        
        # 讀取一個封包
        raw_buffer = sniffer.recvfrom(65565)[0]
        # 從暫存區開頭20bytes建立IP header
        ip_header = IP(raw_buffer[0:20])
      
        #print("Protocol: %s %s -> %s" % (ip_header.protocol, ip_header.src_address, ip_header.dst_address))
    
        # 如果是我們需要的ICMP
        if ip_header.protocol == "ICMP":
            # 計算ICMP封包的開頭位置
            offset = ip_header.ihl * 4
            buf = raw_buffer[offset:offset + sizeof(ICMP)]
            
            # 建立ICMP的結構re
            icmp_header = ICMP(buf)
            
            #print("ICMP -> Type: %d Code: %d" % (icmp_header.type, icmp_header.code))

            # now check for the TYPE 3 and CODE 3 which indicates
            # a host is up but no port available to talk to
            # 確認 TYPE==3  與 CODE==3  表示主機存在但沒有可用的port
            # print("ICMP -> Type: %d Code: %d" % (icmp_header.type, icmp_header.code))
            if icmp_header.code == 3 and icmp_header.type == 3:

                # 確認回應者在我們要掃描的子網路下
                if IPAddress(ip_header.src_address) in IPNetwork(subnet):
                    
                    # 確定封包包含我們的magic_string
                    if raw_buffer[len(raw_buffer)-len(magic_message):] == magic_message:
                        print("Host Up: %s" % ip_header.src_address)
# 處理ctrl-c
except KeyboardInterrupt:
    # 如果使用Windows，關閉混雜模式
    if os.name == "nt":
        sniffer.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)