import sys, os, datetime, traceback
from src.utils.Prints import pt
from src.config.GlobalSettings import LOGGER_PATH


ERROR_LOG = LOGGER_PATH + "errors.log"
API_PETITIONS = LOGGER_PATH + "api_petitions.log"

class Logger:
    separator = "#################################"
    header = separator*2 + "\n" + separator + " START EVENTS " + separator + "\n" + separator*2 + "\n"
    short_header = "----------EVENT---------- " + "\n"
    short_header_error = "----------ERROR---------- " + "\n"

    def __init__(self, error_path=None, writer_path=None):
        global ERROR_LOG, API_PETITIONS
        ERROR_LOG = error_path if error_path else ERROR_LOG
        API_PETITIONS = writer_path if writer_path else API_PETITIONS

    def write_log_error(self, err, info=None, force_path=None):
        exc_type, exc_obj, exc_tb = sys.exc_info()  # this is to get error line number and description.
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]  # to get File Name.
        error_string = "ERROR : Error Msg:{},File Name : {}, Line no : {}\n".format(err, file_name,
                                                                                    exc_tb.tb_lineno)
        pt(error_string)

        if force_path:
            file_log = open(force_path , "a")
        else:
            file_log = open(ERROR_LOG , "a")
        file_log.write(self.short_header_error + str(datetime.datetime.now()) + "\n\n")
        if info:
            file_log.write(str(info) + "\n\n")
        file_log.write(str(err) + "\n\n")

        file_log.close()

    def write_to_logger(self, to_write, starter=False, force_path=None):
        if force_path:
            file_log = open(force_path , "a")
        else:
            file_log = open(API_PETITIONS , "a")
        if starter:
            file_log.write(self.header + str(datetime.datetime.now()) + "\n\n" + str(to_write) + "\n\n")
        else:
            file_log.write(self.short_header + str(datetime.datetime.now()) + "\n\n" + str(to_write) + "\n\n")
        file_log.close()