# -*- coding: utf-8 -*-
"""
Author: @gabvaztor
StartDate: 04/03/2017

This file contains the next information:
    - Libraries to import with installation comment and reason.
    - Data Mining Algorithm.
    - Sets (train,validation and test) information.
    - ANN Arquitectures.
    - A lot of utils methods which you'll get useful advantage


The code's structure is:
    - Imports
    - Global Variables
    - Interface
    - Reading data algorithms
    - Data Mining
    - Training and test
    - Show final conclusions

Style: "Google Python Style Guide"
https://google.github.io/styleguide/pyguide.html

Notes:
    * This file use TensorFlow version >2.0.
"""


def __relative_imports_step_1():
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append('..\\..\\')
    #print("print(sys.path):",sys.path)

import src.config.GlobalSettings as GS
from src.config.Configurator import Configurator
from src.Executor import Executor

def run(user_id=None, model_selected=None):
    if user_id and model_selected:
        Configurator().run()
        Executor(user_id=user_id, model_selected=model_selected).execute()
    else:
        if Configurator().run():
            Executor().execute()

if __name__ == "__main__":
    __relative_imports_step_1()
    print("Executed")
    try:
        Configurator().run_basics()
        GS.LOGGER.write_to_logger("Runner executed")
        # Updating API Mode

        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--userID", required=False,
                        help="userID")
        ap.add_argument("-m", "--userModelSelection", required=False,
                        help="userModelSelection")
        args = vars(ap.parse_args())
        USER_ID = args["userID"] if "userID" in args else None
        MODEL_SELECTED = args["userModelSelection"] if "userModelSelection" in args else None
        if USER_ID and MODEL_SELECTED:
            GS.API_MODE = True
            run(user_id=USER_ID, model_selected=MODEL_SELECTED)  # This means it is a new client petition from PHP.
        else:
            GS.API_MODE = False
            if GS.IS_PREDICTION:
                run(user_id="79.153.245.232_[28-10-2019_16.23.12]", model_selected="retinopathy_k_id")
            else:
                run()
    except Exception as error:
        GS.LOGGER.write_log_error(error)


