"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys

def __relative_imports(number_of_descent):
    sub_folders = "\\.."
    file = __file__
    #print("original file: " + file)
    for _ in range(number_of_descent):
        file = os.path.dirname(file)
        sys.path.append(file)
        #print("file: ", file + sub_folders)
        #sys.path.append(file + sub_folders)
        #sys.path.append(file + sub_folders + "\\")
    root_folder = str(file + "\\")
    root_folder_children = str(file + "..")
    #print("root_folder:", root_folder)
    #print("root_folder_children:", root_folder_children)
    sys.path.append("..")
    #print(sys.path)
    sys.path = list(set(sys.path))
    [print(x) for x in sys.path]

__relative_imports(number_of_descent=4)

import time, datetime,  argparse
from timeit import default_timer as timer

import src.config.GlobalSettings as GS
from src.utils.AsynchronousThreading import object_to_json
from src.utils.Folders import write_string_to_pathfile
from src.utils.Datetimes import date_from_format
from src.utils.Prints import pt
from utils.PetitionObject import Petition, JSON_PETITION_NAME
from src.config.Configurator import Configurator
from src.services.processing.CPrediction import CPrediction

class AnswerConfiguration():
    json_petition_name = JSON_PETITION_NAME
    json_answer_name = "jsonAnswer.json"

    def __init__(self, petition_id, prediction_results=None):
        self.petition_src = PATH_ + "\\" + petition_id + "\\"
        self.model_folder = os.listdir(self.petition_src)[0]
        self.final_petition_dir = self.petition_src + self.model_folder + "\\"
        self.json_petition_src = self.final_petition_dir + self.json_petition_name
        self.json_answer_src = self.final_petition_dir + self.json_answer_name
        self.date = date_from_format(date=datetime.datetime.now())
        self.user_id = USER_ID
        self.user_id_path = USER_ID_PATH
        self.model_selected = MODEL_SELECTED
        prediction_results = self.__get_results(prediction_results)
        if prediction_results:
            if prediction_results.results:
                self.answer = prediction_results.readable_results
            else:
                self.answer = "NOK"
        else:
            self.answer = "NOK"

    def __get_results(self, prediction_results: CPrediction):
        return prediction_results


def execute_clasification(PETITIONS):
    """
    Get petition and classify elements

    Args:
        PETITIONS: List with new petitions

    Returns: petitions_end_ok

    """
    petitions_end_ok = []

    for petition_id in PETITIONS:
        GS.LOGGER.write_to_logger("Petition was found: " + petition_id)
        try:
            # Read petition json
            # TODO (@gabvaztor) Create a different object class to manage paths logic
            path_config = AnswerConfiguration(petition_id=petition_id)
            petition = Petition(path=path_config.json_petition_src, petition_id=petition_id)
            prediction_results = CPrediction(current_petition=petition)
            new_answer_configuration = AnswerConfiguration(petition_id=petition_id,
                                                           prediction_results=prediction_results)
            json_answer_str = object_to_json(object=new_answer_configuration)
            pt(json_answer_str)
            write_string_to_pathfile(string=json_answer_str, filepath=new_answer_configuration.json_answer_src)
            petitions_end_ok.append(petition_id)
            GS.LOGGER.write_to_logger("Petition finished")
        except Exception as error:
            import traceback
            traceback.print_exc()
            GS.LOGGER.write_log_error(error)

    return petitions_end_ok

def __get_new_online_petitions():
    global PETITIONS

    # First time
    start = timer()
    #past_petitions = __get_new_folders(petitions=PETITIONS)
    petitions_counts = 0
    sleeps_counts = 0
    while True:
        #pt("p1", past_petitions)
        PETITIONS = __get_new_folders(petitions=PETITIONS)
        #pt("p2", PETITIONS)
        #PETITIONS = list(set(PETITIONS) - set(past_petitions))
        if PETITIONS:
            pt("\n")
            pt("Petitions:", PETITIONS, "|@@| Date:[" + str(date_from_format(date=datetime.datetime.now()) + "]"))
            pt("\n")
        elif sleeps_counts % 10 == 0:
            pt("Total Counts: " + str(petitions_counts) + " ### Petitions:", PETITIONS, "|@@| Date:[" + str(date_from_format(date=datetime.datetime.now()) + "]"), same_line=True)
            #if sleeps_counts % 600: gc.collect()
        if PETITIONS:
            execute_clasification(PETITIONS)
            # TODO Detele folders
            # TODO if classification OK or timeout, then move/delete folder petition
            #past_petitions = past_petitions + petitions_end_ok
            #PETITIONS = list(set(PETITIONS) - set(petitions_end_ok))
            petitions_counts += 1

            sys.exit()
            exit()
            quit()


        end = timer()
        if end - start >= 600:
            exit()
            quit()
            sys.exit()
        time.sleep(0.2)
        sleeps_counts += 1

def __get_new_folders(petitions):
    """
    Returns:

    """
    users_ids = os.listdir(PATH_)
    if USER_ID in users_ids:
        users_ids = [USER_ID]
    else:
        users_ids.clear()
    return users_ids

def run():
    __get_new_online_petitions()


try:
    # Example:
    # python "../API.py"  -i 79.153.245.232_[29-10-2019_14.34.19] -m retinopathy_k_id
    Configurator().run_basics()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--userID", required=False,
                    help="userID")
    ap.add_argument("-m", "--userModelSelection", required=False,
                    help="userModelSelection")

    args = vars(ap.parse_args())

    GS.LOGGER.write_to_logger("API executed")
    USER_ID = args["userID"] if "userID" in args else ""
    MODEL_SELECTED = args["userModelSelection"] if "userModelSelection" in args else ""

    PETITIONS = []
    TRIES = 0
    PATH_ = r"Z:\Data_Science\Conciences\Framework\Uploads"
    USER_ID_PATH = PATH_ + "\\" + USER_ID if USER_ID else PATH_ + "\\"
    __get_new_online_petitions()

except Exception as e:
    USER_ID = ""  # To avoid warning
    GS.LOGGER.write_log_error(e)
    sys.exit()

if __name__ == "__main__":

    #__relative_imports_step_1()
    try:
        # Example:
        # python "..\API.py" -i 79.153.245.232_[29-10-2019_14.34.19] -m retinopathy_k_id
        Configurator().run_basics()
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--userID", required=False,
                        help="userID")
        ap.add_argument("-m", "--userModelSelection", required=False,
                        help="userModelSelection")

        args = vars(ap.parse_args())

        GS.LOGGER.write_to_logger("API executed")
        USER_ID = args["userID"] if "userID" in args else ""
        MODEL_SELECTED = args["userModelSelection"] if "userModelSelection" in args else ""

        PETITIONS = []
        TRIES = 0
        PATH_ = r"Z:\Data_Science\Conciences\Framework\Uploads"
        USER_ID_PATH = PATH_ + "\\" + USER_ID if USER_ID else PATH_ + "\\"
        __get_new_online_petitions()

    except Exception as e:
        USER_ID = ""  # To avoid warning
        GS.LOGGER.write_log_error(e)
        sys.exit()



