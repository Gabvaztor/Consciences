import sys, os, datetime, traceback
from src.utils.Prints import pt
from src.config.GlobalSettings import LOGGER_PATH


ERROR_LOG = LOGGER_PATH + "errors.log"
API_PETITIONS = LOGGER_PATH + "api_petitions.log"

class Logger:
    separator = "#################################"
    header = separator*2 + "\n" + separator + " START EVENTS " + separator + "\n" + separator*2 + "\n"
    short_header = "EVENT " + "\n"

    def write_log_error(self, err):
        exc_type, exc_obj, exc_tb = sys.exc_info()  # this is to get error line number and description.
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]  # to get File Name.
        error_string = "ERROR : Error Msg:{},File Name : {}, Line no : {}\n".format(err, file_name,
                                                                                    exc_tb.tb_lineno)
        pt(error_string)

        file_log = open(ERROR_LOG , "a")
        file_log.write(error_string + "\n\n" + str(err) + "\n\n")
        file_log.close()

    def write_to_logger(self, to_write, starter=False):
        file_log = open(API_PETITIONS , "a")
        if starter:
            file_log.write(self.header + str(datetime.datetime.now()) + "\n\n" + str(to_write) + "\n\n")
        else:
            file_log.write(self.short_header + str(datetime.datetime.now()) + "\n\n" + str(to_write) + "\n\n")
        file_log.close()