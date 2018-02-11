# 引用標準函式庫中的getpass模組  使密碼輸入不會是明碼顯示
import getpass
import os
import socket
import select
try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

import sys
# 處理使用者輸入command line的參數之模組 optparse(在python 3.2 版本後被棄用 建議改用argparse模組)
from optparse import OptionParser

import paramiko
# 自訂ssh使用的port
SSH_PORT = 22
# 預設的prot
DEFAULT_PORT = 4000
#是否顯示除錯訊息的全域布林變數 (True為顯示 False為不顯示)
g_verbose = True


class ForwardServer (SocketServer.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    
# 啟動新的執行序來處理連線
class Handler (SocketServer.BaseRequestHandler):

    def handle(self):
        try:
            chan = self.ssh_transport.open_channel('direct-tcpip',
                                                   (self.chain_host, self.chain_port),
                                                   self.request.getpeername())
        except Exception as e:
            verbose('Incoming request to %s:%d failed: %s' % (self.chain_host,
                                                              self.chain_port,
                                                              repr(e)))
            return
        if chan is None:
            verbose('Incoming request to %s:%d was rejected by the SSH server.' %
                    (self.chain_host, self.chain_port))
            return

        verbose('Connected!  Tunnel open %r -> %r -> %r' % (self.request.getpeername(),
                                                            chan.getpeername(), (self.chain_host, self.chain_port)))
        while True:
            r, w, x = select.select([self.request, chan], [], [])
            if self.request in r:
                data = self.request.recv(1024)
                if len(data) == 0:
                    break
                chan.send(data)
            if chan in r:
                data = chan.recv(1024)
                if len(data) == 0:
                    break
                self.request.send(data)
                
        peername = self.request.getpeername()
        chan.close()
        self.request.close()
        verbose('Tunnel closed from %r' % (peername,))


def forward_tunnel(local_port, remote_host, remote_port, transport):
    # this is a little convoluted, but lets me configure things for the Handler
    # object.  (SocketServer doesn't give Handlers any way to access the outer
    # server normally.)
    class SubHander (Handler):
        chain_host = remote_host
        chain_port = remote_port
        ssh_transport = transport
    ForwardServer(('', local_port), SubHander).serve_forever()

# 自訂顯示除錯訊息之函式 若全域變數 g_verbose = True 則顯示訊息
def verbose(s):
    if g_verbose:
        print(s)


HELP = """\
Set up a forward tunnel across an SSH server, using paramiko. A local port
(given with -p) is forwarded across an SSH session to an address:port from
the SSH server. This is similar to the openssh -L option.
"""

# 由字串解析出主機ip與port
def get_host_port(spec, default_port):
    "parse 'hostname:22' into a host and port, with the port optional"
    args = (spec.split(':', 1) + [default_port])[:2]
    args[1] = int(args[1])
    return args[0], args[1]

# 解析使用者輸入的參數
def parse_options():
    global g_verbose
    
    parser = OptionParser(usage='usage: %prog [options] <ssh-server>[:<server-port>]',
                          version='%prog 1.0', description=HELP)
    parser.add_option('-q', '--quiet', action='store_false', dest='verbose', default=True,
                      help='squelch all informational output')
    parser.add_option('-p', '--local-port', action='store', type='int', dest='port',
                      default=DEFAULT_PORT,
                      help='local port to forward (default: %d)' % DEFAULT_PORT)
    parser.add_option('-u', '--user', action='store', type='string', dest='user',
                      default=getpass.getuser(),
                      help='username for SSH authentication (default: %s)' % getpass.getuser())
    parser.add_option('-K', '--key', action='store', type='string', dest='keyfile',
                      default=None,
                      help='private key file to use for SSH authentication')
    parser.add_option('', '--no-key', action='store_false', dest='look_for_keys', default=True,
                      help='don\'t look for or use a private key file')
    parser.add_option('-P', '--password', action='store_true', dest='readpass', default=False,
                      help='read password (for key or password auth) from stdin')
    parser.add_option('-r', '--remote', action='store', type='string', dest='remote', default=None, metavar='host:port',
                      help='remote host and port to forward to')
    options, args = parser.parse_args()
    # 參數錯誤顯示的訊息
    if len(args) != 1:
        parser.error('Incorrect number of arguments.')
    # 若必代的參數未填時所顯示的訊息
    if options.remote is None:
        parser.error('Remote address required (-r).')
    # 是否顯示除錯訊息 (根據使用者填入的verbose變數)
    g_verbose = options.verbose
    # 讀取伺服器的ip與port
    server_host, server_port = get_host_port(args[0], SSH_PORT)
    # 讀取遠端主機的ip與port
    remote_host, remote_port = get_host_port(options.remote, SSH_PORT)
    # 回傳三個變數:使用者輸入的選項、(伺服器ip, 伺服器port)、(遠端主機ip, 遠端主機port)
    return options, (server_host, server_port), (remote_host, remote_port)

# 主函式
def main():
    # 解析使用者的選項與 伺服器 與 遠端主機
    options, server, remote = parse_options()
    # 密碼預設為None
    password = None
    # 如果選項中需要使用者輸入ssh密碼 則利用getpass模組來讀取ssh的密碼
    if options.readpass:
        password = getpass.getpass('Enter SSH password: ')
    
    # 建立ssh的Client端
    client = paramiko.SSHClient()
    #client.load_system_host_keys()
    # 使用傳統的使用者名稱與密碼的方式進行認證
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    # 顯示連線伺服器的訊息
    verbose('Connecting to ssh host %s:%d ...' % (server[0], server[1]))
    # 嘗試發起連線
    try:
        # 發起連線
        client.connect(server[0], server[1], username=options.user, key_filename=options.keyfile,
                       look_for_keys=options.look_for_keys, password=password)
    except Exception as e:
        # 由於ecption為異常現象 因此固定使用print來顯示
        # print('*** Failed to connect to %s:%d: %r' % (server[0], server[1], e))
        verbose('*** Failed to connect to %s:%d: %r' % (server[0], server[1], e))
        sys.exit(1)
    # 除錯訊息
    verbose('Now forwarding port %d to %s:%d ...' % (options.port, remote[0], remote[1]))

    try:
        forward_tunnel(options.port, remote[0], remote[1], client.get_transport())
    except KeyboardInterrupt:
        print('C-c: Port forwarding stopped.')
        sys.exit(0)


if __name__ == '__main__':
    main()