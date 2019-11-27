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
import select

class Server():

    def __init__(self, port=None, host=None):
        if not port or not host:
            self.host = socket.gethostname()
            self.host = HOST_SERVER
            self.port = SERVER_PORT
        self.counts_petitions = 0
        self.sockets_list = []
        self.connect()
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
            print("Server created and listening at address: " + str(self.host) + ":" + str(self.port))
            self.sockets_list.append(self.tcp_socket)
        except Exception as error:
            print(str(error))
            LOGGER.write_log_error(error, str(error))
            return None

    def handle_requests(self):
        """
        This method tells python what to do when a request is made on the given port.
        """

        self.client_sockets_queue = {}
        if self.tcp_socket:
            while True:
                try:
                    client_socket, cliend_address = self.tcp_socket.accept()
                except Exception as error:
                    LOGGER.write_log_error(error, str(error))
                    client_socket, cliend_address = [(x,y) for x,y in self.client_sockets_queue.items()]
                if client_socket and cliend_address:
                    self.client_sockets_queue.update({client_socket : cliend_address})
                    LOGGER.write_to_logger("A connection has been detected from dir: " + str(cliend_address))
                    print("---------------------------------------------\n"
                          "A connection has been detected from dir: " + str(cliend_address))
                    msg_ = "[SERVER] I am here!" + "\n"
                    self.send_message_to_client(client_socket=client_socket, message=msg_)
                    self.counts_petitions += 1
                    print("Client found!. Total petitions: " + str(self.counts_petitions))
                    if self.receive_message(client_socket=client_socket):
                        #option_selected = str(input("Select an option to send to server"))
                        self.send_message_to_client(client_socket=client_socket, message="second message")
                        #self.tcp_socket.close()

        else:
            raise NotImplementedError


    def handle_requests_v2(self):
        """
        This method tells python what to do when a request is made on the given port.
        """
        self.client_sockets_queue = {}
        if self.tcp_socket:
            while True:
                read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)
                for n_socket in read_sockets:
                    if n_socket == self.tcp_socket:
                        try:
                            client_socket, client_address = self.tcp_socket.accept()
                        except Exception as error:
                            LOGGER.write_log_error(error, str(error))
                            client_socket, client_address = [(x,y) for x,y in self.client_sockets_queue.items()]
                        if client_socket and client_address:
                            self.client_sockets_queue.update({client_socket : client_address})
                            LOGGER.write_to_logger("A connection has been detected from dir: " + str(client_address))

                            current_client = self.receive_message(client_socket=client_socket)
                            if not current_client:
                                continue
                            self.sockets_list.append(client_socket)  # We need to add the client to the asynchronous list
                            self.client_sockets_queue[client_socket] = current_client
                            self.counts_petitions += 1
                            print(f"Client found!. Total petitions: {str(self.counts_petitions)}\nFrom address: "
                                  f"{client_address[0]} username:{current_client['data']}")
                            msg_ = "[SERVER] I am here!" + "\n"
                            self.send_message_to_client(client_socket=client_socket, message=msg_)
                    else:
                        message = self.receive_message(client_socket=n_socket)
                        if not message:
                            print(f"Closed connection from username:{n_socket['data']}")
                            self.sockets_list.remove(n_socket)
                            del self.client_sockets_queue[n_socket]
                            continue
                        else:
                            user = self.client_sockets_queue[n_socket]
                            print(f"Received message from {user['data']}: {message['data']}")

                        for client_socket in self.sockets_list:
                            if client_socket != n_socket:
                                client_socket.send(user["header"] + user["data"] + message["header"] + message["data"])
                for n_socket in exception_sockets:
                    self.sockets_list.remove(n_socket)
                    del self.client_sockets_queue[n_socket]

                """
                if self.receive_message(client_socket=client_socket):
                    #option_selected = str(input("Select an option to send to server"))
                    self.send_message_to_client(client_socket=client_socket, message="second message")
                    #self.tcp_socket.close()
                """


        else:
            raise NotImplementedError

    def send_message_to_client(self, client_socket, message):
        client_socket.send(bytes(message,"utf-8"))

    def receive_message(self, client_socket):
        try:
            msg = client_socket.recv(HEADER_LENGTH)
            if len(msg) > 0:
                message = msg.decode("utf-8")
                LOGGER.write_to_logger(message)
                print(message)
                message_length = int(message.strip())
                return {"header": msg, "data": client_socket.recv(message_length).decode("utf-8")}
            else:
                print("No message found")
                LOGGER.write_to_logger("No message found")
                return False
        except Exception as error:
            return False

if __name__ == "__main__":
    __get_root_project(number_of_descent=4)
    from src.examples.sockets.Parameters import HOST_SERVER, SERVER_PORT, ERROR_LOGGER_SERVER_PATH, LOGGER_SERVER_PATH,\
        HEADER_LENGTH
    from src.utils.Logger import Logger
    LOGGER = Logger(writer_path=LOGGER_SERVER_PATH, error_path=ERROR_LOGGER_SERVER_PATH)
    SERVER = Server()