"""
python Z:\Data_Science\Projects\Framework_API_Consciences\src\sockets\Server.py
"""

def __get_root_project(number_of_descent):
    import sys, os
    file = __file__
    for _ in range(number_of_descent):
        file = os.path.dirname(file)
        sys.path.append(file)

import socket

class Server():

    def __init__(self, port=None, host=None):
        if not port or not host:
            self.host = HOST_SERVER
            self.port = SERVER_PORT
        self.counts_petitions = 0
        self.tcp_socket = self.connect()
        self.handle_requests()


    def connect(self):
        """
        Create a socket and bind connection. Also, makes it listener.
        """
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # As TCP
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # socket options
            self.tcp_socket.bind((self.host, self.port))  # Create the bind
            self.tcp_socket.listen()  # Make it a constant listener for X clients (as queue)
            print("Server created and listening")
            return self.tcp_socket
        except Exception as error:
            print(str(error))
            LOGGER.write_log_error(error, str(error))
            return None

    def handle_requests(self):
        """
        This method tells python what to do when a request is made on the given port.
        """
        if self.tcp_socket:
            while True:
                csock, caddr = self.tcp_socket.accept()
                LOGGER.write_to_logger("A connection has been detected from dir: " + str(caddr))
                msg_ = "[SERVER] I am here!" + "\n"
                csock.send(bytes(msg_,"utf-8"))
                self.counts_petitions += 1
                print("Client found!. Total petitions: " + str(self.counts_petitions))
                another = input()
                if another == "STOP":
                    break
        else:
            raise NotImplementedError

if __name__ == "__main__":
    __get_root_project(number_of_descent=4)
    from src.examples.sockets.Parameters import HOST_SERVER, SERVER_PORT, ERROR_LOGGER_SERVER_PATH, LOGGER_SERVER_PATH
    from src.utils.Logger import Logger
    LOGGER = Logger(writer_path=LOGGER_SERVER_PATH, error_path=ERROR_LOGGER_SERVER_PATH)
    SERVER = Server()