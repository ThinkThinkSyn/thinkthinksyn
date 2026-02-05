# -*- coding: utf-8 -*-
'''When running in local mode, all connections will be forwarded to remote servers.'''
if __name__ == "__main__": # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'thinkthinksyn.common_utils.network_utils'

import os
import re
import atexit
import logging

from pathlib import Path
from functools import cache
from paramiko import SSHConfig
from sshtunnel import SSHTunnelForwarder

from .helper_funcs import get_available_port

def _get_env(key, default=None)->str|None:
    return os.environ.get(key, default)

_logger = logging.getLogger(__name__)

_remote_tunnels: dict[tuple[str, str, int, int|None], SSHTunnelForwarder] = {}   # {(ssh_ip, remote_ip, remote_port, target_local_port): SSHTunnelForwarder}
_LOCALHOST = '127.0.0.1'
try:
    _DEFAULT_SSH_PORT = _get_env('SSH_PORT', '22')  # type: ignore
    _DEFAULT_SSH_PORT = int(_DEFAULT_SSH_PORT)  # type: ignore  
except ValueError:
    _DEFAULT_SSH_PORT: int = 22
_DEFAULT_SSH_USERNAME = _get_env('SSH_USERNAME') or 'root'

_ssh_full_pattern = re.compile(r'^(?:(?P<user>[^:@]+)(?::(?P<pw>[^@]+))?@)?(?P<ip>[^:]+)(?::(?P<port>\d+))?$')

def _get_default_ssh_key_path() -> str|None:
    if keypath:= _get_env('SSH_KEY_PATH'):
        if os.path.exists(keypath):
            return keypath
    if key:= _get_env('SSH_KEY'):
        count = 0
        while os.path.exists(os.path.join(os.path.expanduser("~"), ".ssh", f".tts_temp_ssh_key_{count}")):
            count += 1
        temp_key_path = os.path.join(os.path.expanduser("~"), ".ssh", f".tts_temp_ssh_key_{count}")
        with open(temp_key_path, 'w') as f:
            f.write(key)
        os.chmod(temp_key_path, 0o600)
        return temp_key_path
    home = os.path.expanduser("~")
    default_key = os.path.join(home, ".ssh", "id_rsa")
    if os.path.exists(default_key):
        return default_key
    default_key = os.path.join(home, ".ssh", "id_ed25519")
    if os.path.exists(default_key):
        return default_key
    return None

@cache
def _get_default_ssh_config():
    ssh_config_path = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(ssh_config_path):
        return None
    with open(ssh_config_path) as f:
        config = SSHConfig()
        try:
            config.parse(f)
            return config
        except Exception as e:
            _logger.warning(f'Failed to parse SSH config file. {type(e).__name__}: {e}')
            return None

def _find_ssh_config(ssh_ip: str)->tuple[int, str, str|None]|None:    # (port, username, key_path)
    # find ssh config from .ssh/config file
    config = _get_default_ssh_config()
    if not config:
        return None
    host_config = config.lookup(ssh_ip)
    if not host_config:
        return None
    if len(host_config) ==1 and 'hostname' in host_config:
        return None
    try:
        port = int(host_config.get('port', _DEFAULT_SSH_PORT))  # type: ignore
    except:
        port = _DEFAULT_SSH_PORT
    username = host_config.get('user', _DEFAULT_SSH_USERNAME)
    key_path = host_config.get('identityfile', [None])[0]
    if not key_path:
        key_path = _get_default_ssh_key_path()
    return port, username, key_path

def _start_ssh_tunnel_server(
    ssh_ip:str,
    ssh_remote_port:int,
    ssh_remote_ip:str=_LOCALHOST,
    ssh_port:int|None=None,
    ssh_user:str|None=None,
    ssh_pw: str|None=None,
    ssh_key_path: str|Path|None=None,
    local_port:int|None=None
):  # type: ignore
    if not ssh_port or not ssh_user or (not ssh_key_path or not ssh_pw):
        if config:=_find_ssh_config(ssh_ip):
            ssh_port_config, ssh_user_config, ssh_key_path_config = config
            if not ssh_port:
                ssh_port = ssh_port_config
            if not ssh_user:
                ssh_user = ssh_user_config
            if not ssh_key_path and not ssh_pw:
                ssh_key_path = ssh_key_path_config
        else:
            ssh_port = ssh_port or _DEFAULT_SSH_PORT
            ssh_user = ssh_user or _DEFAULT_SSH_USERNAME
            if not ssh_key_path and not ssh_pw:
                ssh_key_path = _get_default_ssh_key_path()
    
    if not ssh_key_path and not ssh_pw:
        if env_pw:= _get_env('SSH_PW'):
            ssh_pw = env_pw
        else:
            raise ValueError('No SSH key path or password provided, and none found in SSH config or environment variables.')
    
    if tunnel:=_remote_tunnels.get((ssh_ip, ssh_remote_ip, ssh_port, local_port)):
        return tunnel.local_bind_port
    
    target_port = local_port if local_port is not None else get_available_port()
    t = SSHTunnelForwarder(
        ssh_address_or_host=(ssh_ip, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_pw,
        ssh_pkey=ssh_key_path,
        remote_bind_address=(ssh_remote_ip, ssh_remote_port),
        local_bind_address=(_LOCALHOST, target_port),
    )
    _remote_tunnels[(ssh_ip, ssh_remote_ip, ssh_port, local_port)] = t
    t.start()
    return target_port

def stop_remote_clients():
    for client in tuple(_remote_tunnels.values()):
        client.stop()

atexit.register(stop_remote_clients)
    
def get_remote_forward_tunnel(
    ssh_ip_or_host:str,
    ssh_remote_port:int,
    ssh_remote_ip:str=_LOCALHOST,
    ssh_port:int|None=None,
    ssh_user:str|None=None,
    ssh_pw: str|None=None,
    ssh_key_path: str|Path|None=None, 
    local_port:int|None=None
)->int: # type: ignore
    '''
    Create or get a port forwarding to the given remote ip and remote port.
    In non-local-mode(which means you are running in server), this function will do nothing. In that case,
    it will directly return the given remote ip (remote ip will change to 127.0.0.1 if it is the same of 
    this server) and port.
    
    Args:
        ssh_ip_or_host: ssh ip or host to connect to. If user & pw seems to included in the ip string, they will be parsed out.
                    If it can be found in ssh config file(~/.ssh/config), the port, username and key path will also be parsed out.
        ssh_remote_port: remote port to forward to.
        ssh_remote_ip: remote ip to forward to. Default to `localhost` of the remote server.
        ssh_port: remote port to forward to. 
        ssh_user: ssh username to connect to the remote server.
        ssh_pw: ssh password to connect to the remote server.
        ssh_key_path: ssh private key path to connect to the remote server.
        local_port: if given, will try to forward to this port, if not given, will find an available port to forward to.
    
    Available env variables:
        SSH_PORT: default ssh port to connect to.
        SSH_USERNAME: default ssh username to connect to.
        SSH_KEY_PATH: default ssh private key path to connect to.
        SSH_KEY: default ssh private key content to connect to.
        SSH_PW: default ssh password to connect to.
    
    Returns:
        The final local port used for forwarding.
    '''
    if not ssh_ip_or_host:
        raise ValueError(f'Invalid remote ip or host: `{ssh_ip_or_host}`')
    
    if m:= _ssh_full_pattern.match(ssh_ip_or_host):
        user = m.group('user')
        pw = m.group('pw')
        ip = m.group('ip')
        port_str = m.group('port')
        if user and not ssh_user:
            ssh_user = user
        if pw and not ssh_pw:
            ssh_pw = pw
        ssh_ip_or_host = ip
        if port_str and not ssh_port:
            try:
                ssh_port = int(port_str)
            except:
                pass
    if not ssh_ip_or_host:
        raise ValueError(f'Invalid remote ip: {ssh_ip_or_host}')
    
    local_port_used = _start_ssh_tunnel_server(
        ssh_ip=ssh_ip_or_host,
        ssh_remote_port=ssh_remote_port,
        ssh_remote_ip=ssh_remote_ip,
        ssh_port=ssh_port,
        ssh_user=ssh_user,
        ssh_pw=ssh_pw,
        ssh_key_path=ssh_key_path,
        local_port=local_port
    )
    return local_port_used


__all__ = ['get_remote_forward_tunnel', 'stop_remote_clients']
    