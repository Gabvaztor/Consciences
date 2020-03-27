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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from platform import python_version
print("Python version:", python_version())

def __get_root_project(number_of_descent):
    import sys, os
    file = __file__
    for _ in range(number_of_descent):
        file = os.path.dirname(file)
        sys.path.append(file)
    sys.path = list(set(sys.path))

if __name__ == "__main__":
    __get_root_project(number_of_descent=3)

import src.config.GlobalSettings as GS
from src.config.Configurator import Configurator
from src.Executor import Executor

def run():
    if Configurator().run():
        Executor().execute()

if __name__ == "__main__":
    print("Runner Executed")
    GS.API_MODE = False
    GS.IS_PREDICTION = False
    try:
        Configurator().run_basics()
        GS.LOGGER.write_to_logger("Runner executed", starter=True)
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
            print("USER_ID", USER_ID)
            print("MODEL_SELECTED", MODEL_SELECTED)
            try:
                Executor(user_id=USER_ID, model_selected=MODEL_SELECTED).execute()
                  # This means it is a new client petition from PHP.
            except Exception as e:
                GS.LOGGER.write_log_error(e)
                print("Error", e)
        else:
            GS.API_MODE = False
            if GS.IS_PREDICTION:
                Executor(user_id="79.153.245.232_[28-10-2019_16.23.12]",
                         model_selected="retinopathy_k_id").execute()
            else:
                run()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print("USER_ID", 2)
        print("MODEL_SELECTED", 3)
        GS.LOGGER.write_log_error(error)


