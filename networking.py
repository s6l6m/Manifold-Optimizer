import socket

def get_free_port(host: str = "0.0.0.0") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def get_url_for_port(port: int, host: str = "localhost", protocol: str = "http") -> str:
    return f"{protocol}://{host}:{port}"


def check_port_in_use(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
