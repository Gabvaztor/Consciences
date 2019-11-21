import os

CURRENT_DIR = os.path.dirname(__file__)
LOGS_DIR = CURRENT_DIR + r"\logs"
LOGGER_SERVER_PATH = LOGS_DIR + r"\server.log"
ERROR_LOGGER_SERVER_PATH = LOGS_DIR + r"\server_error.log"
LOGGER_CLIENT_PATH = LOGS_DIR + r"\client.log"
ERROR_LOGGER_CLIENT_PATH = LOGS_DIR + r"\client_error.log"

SERVER_PORT = 5056
HOST_SERVER = "127.0.0.1"
PUBLIC_IP = ""
PUBLIC_IP_ = ""