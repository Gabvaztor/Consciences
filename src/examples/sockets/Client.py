"""
python Z:\Data_Science\Projects\Consciences\src\examples\sockets\Client.py
"""

def __get_root_project(number_of_descent):
    import sys, os
    file = __file__
    for _ in range(number_of_descent):
        file = os.path.dirname(file)
        sys.path.append(file)

import socket
import time
class Client():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((HOST_SERVER, SERVER_PORT))
        print("Connected to host: " + str(self.host))
        attemps = 0
        answered = False
        while True:
            try:
                print("Attemp number: " + str(attemps+1))
                if attemps >= 5:
                    LOGGER.write_to_logger("Attemp number: " + str(attemps))
                    if answered:
                        print("Listener by a limit of 5 seconds: STOP")
                    else:
                        print("No answer from server in 5 seconds: STOP")
                    self.tcp_socket.close()
                    break
                msg = self.tcp_socket.recv(1024)
                if len(msg) > 0:
                    LOGGER.write_to_logger(msg.decode("utf-8"))
                    print(msg.decode("utf-8"))
                    answered = True
                    self.tcp_socket.close()
                else:
                    LOGGER.write_to_logger("No message found")
                time.sleep(1)  # One second per attemp
                attemps += 1
                LOGGER.write_to_logger("Attemp number: " + str(attemps))
                if answered:
                    break
            except Exception as error:
                LOGGER.write_log_error(error, str(error))
                break

if __name__ == "__main__":
    __get_root_project(number_of_descent=4)
    from src.examples.sockets.Parameters import SERVER_PORT, HOST_SERVER, PUBLIC_IP, ERROR_LOGGER_CLIENT_PATH, \
        LOGGER_CLIENT_PATH
    from src.utils.Logger import Logger
    LOGGER = Logger(writer_path=LOGGER_CLIENT_PATH, error_path=ERROR_LOGGER_CLIENT_PATH)
    CLIENT = Client(port=SERVER_PORT, host=PUBLIC_IP)