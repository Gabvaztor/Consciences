"""

"""
import os, sys, time, datetime,  argparse
from timeit import default_timer as timer

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../../')

from src.utils.AsynchronousThreading import object_to_json
from src.utils.Folders import write_string_to_pathfile
from src.utils.Datetimes import date_from_format
from src.utils.Prints import pt

from src.services.processing.CPrediction import CPrediction

try:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--userID", required=False,
                    help="userID")
    ap.add_argument("-m", "--userModelSelection", required=False,
                    help="userModelSelection")

    args = vars(ap.parse_args())

    USER_ID = args["userID"] if "userID" in args else ""
    MODEL_SELECTED = args["userModelSelection"] if "userModelSelection" in args else ""

except Exception as e:
    USER_ID = ""  # To avoid warning
    sys.exit()

PETITIONS = []
TRIES = 0
PATH_ = r"Z:\Data_Science\Conciences\Framework\Uploads"
USER_ID_PATH = PATH_ + "\\" + USER_ID if USER_ID else PATH_ + "\\"

class AnswerConfiguration():
    json_petition_name = "jsonPetition.json"
    json_answer_name = "jsonAnswer.json"

    def __init__(self, petition_id):
        self.petition_src = PATH_ + "\\" + petition_id + "\\"
        self.model_folder = os.listdir(self.petition_src)[0]
        self.final_petition_dir = self.petition_src + self.model_folder + "\\"
        self.json_src = self.final_petition_dir + self.json_petition_name
        self.json_answer_src = self.final_petition_dir + self.json_answer_name
        self.answer = "OK"
        self.date = date_from_format(date=datetime.datetime.now())
        self.user_id = USER_ID
        self.user_id_path = USER_ID_PATH
        self.model_selected = MODEL_SELECTED

def execute_clasification(PETITIONS):
    """
    Get petition and classify elements

    Args:
        PETITIONS: List with new petitions

    Returns: petitions_end_ok

    """
    petitions_end_ok = []

    for petition_id in PETITIONS:
        try:
            new_answer_configuration = AnswerConfiguration(petition_id=petition_id)
            new_prediction = CPrediction(answer_configuration=new_answer_configuration)
            json_answer_str = object_to_json(object=new_answer_configuration)
            pt(json_answer_str)
            write_string_to_pathfile(string=json_answer_str, filepath=new_answer_configuration.json_answer_src)
            petitions_end_ok.append(petition_id)
        except Exception as e:
            pt(e)

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


if __name__ == "__main__":
    #execute_asynchronous_thread(__get_new_online_petitions)
    run()

