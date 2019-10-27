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
    * This file use TensorFlow version >1.0.
"""

"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""

'''LOCAL IMPORTS
* UtilsFunctions is a library that contains a lot of functions which will help us
to code expressively, clearly and efficiently.
* TensorFlowGUI's library contains all GUI's methods. Contains EasyGUI.
Here you can download the library: https://pypi.python.org/pypi/easygui#downloads
It had been used the version: 0.98.1
'''
import os
import sys
import importlib

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('../../')

import src.services.preparation.CCReader as tfr
from src.config.Projects import Projects
from src.utils.Dictionary import Dictionary
from src.utils.Prints import pt
from src.config.GlobalSettings import PROBLEM_ID, PROJECT_ROOT_PATH, IS_PREDICTION
from src.services.modeling.CModels import CModels

''' TensorFlow: https://www.tensorflow.org/
To upgrade TensorFlow to last version:
*CPU: pip3 install --upgrade tensorflow
*GPU: pip3 install --upgrade tensorflow-gpu
'''
# import tensorflow as tf
# print("TensorFlow: " + tf.__version__)


''' Numpy is an extension to the Python programming language, adding support for large,
multi-dimensional arrays and matrices, along with a large library of high-level
mathematical functions to operate on these arrays.
It is mandatory to install 'Numpy+MKL' before scipy.
Install 'Numpy+MKL' from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
http://www.numpy.org/
https://en.wikipedia.org/wiki/NumPy '''
import numpy as np

'''
# You need to install the 64bit version of Scipy, at least on Windows.
# It is mandatory to install 'Numpy+MKL' before scipy.
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
# We can find scipi in the url: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy'''

''' Matlab URL: http://matplotlib.org/users/installing.html
python -m pip3 install matplotlib'''

''' TFLearn library. License MIT.
Git Clone : https://github.com/tflearn/tflearn.git
    To install: pip3 install tflearn'''
'''
 Sklearn(scikit-learn): Simple and efficient tools for data mining and data analysis
 To install: pip3 install -U scikit-learn
'''
"""
To install pandas: pip3 install pandas
"""

PROJECT_ID_PACKAGE = "src.projects." + PROBLEM_ID
MODELING_PACKAGE = PROJECT_ID_PACKAGE + ".modeling"
MODULE_NAME = ".Models"
MODULE_CONFIG = ".Config"
SETTING_OBJECT = Projects.get_settings()

class Executor:

    def __init__(self, user_id=None, model_selected=None):
        self.user_id = user_id
        self.model_selected = model_selected

    def execute(self):
        if self.user_id and self.model_selected:  # If is not None, it is a Petition
            _api_process(user_id=self.user_id, model_selected=self.model_selected,
                         petition_process_in_background=True)
        else:

            _core_process()

def _api_process(user_id: str, model_selected: str, petition_process_in_background=True):
    """
    Execute api process in background (optional)
    Args:
        user_id: user id sent from PHP server/client
        petition_process_in_background: If the process will be executed in the background
    """
    if petition_process_in_background:
        import subprocess
        import src.services.api.API as api

        try:  # Getting fullpath from api module
            filepath = PROJECT_ROOT_PATH + str(api.__file__)
            pt("filepath", filepath)
            bat_path = "Z:\\Data_Science\\Projects\\Framework_API_Consciences\\src\\AIModels_FW_main.bat"
            python_path = 'python "Z:\Data_Science\Projects\Framework_API_Consciences\src\MainLoop.py"' + " -i " + user_id
            python_path = 'python ' + filepath + " -i " + user_id + " -m " + model_selected
            #p = subprocess.Popen(bat_path, creationflags=subprocess.CREATE_NEW_CONSOLE)
            subprocess.Popen(python_path, creationflags=subprocess.CREATE_NEW_CONSOLE)
        except:
            pass

def _core_process():

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ---- GLOBAL VARIABLES ----
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    """

    """
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ---- USER INTERFACE ----
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    """Creating user interface
    #properties = eg.EasyGui()
    #uf.pt("Typos GUI",properties.types)
    
    """
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ---- READING DATA ----
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    """
    Creating Reader Features
    """
    OPTION_PROBLEM = Projects.retinopathy_k_problem_id
    options = [OPTION_PROBLEM, 1, 720, 1280]
    path_train_and_test_images = [SETTING_OBJECT.train_path, SETTING_OBJECT.test_path]
    number_of_classes = 5  # Start in 0
    percentages_sets = None  # Example
    labels_set = [Dictionary.string_labels_type_option_hierarchy]
    is_an_unique_csv = False  # If this variable is true, then only one CSV file will be passed and it will be treated like
    # trainSet, validationSet(if necessary) and testSet
    known_data_type = ''  # Contains the type of data if the data file contains an unique type of data. Examples: # Number
    # or Chars.

    # TODO (@gabvaztor) Check if file exist autom<atically
    load_dataset = True
    if load_dataset:
        path_to_load = SETTING_OBJECT.saved_dataset_path
        x_train_string = "x_train.npy"
        y_train_string = "y_train.npy"
        x_test_string = "x_test.npy"
        y_test_string = "y_test.npy"

        x_train = np.load(file=path_to_load + x_train_string)
        y_train = np.load(file=path_to_load + y_train_string)
        x_test = np.load(file=path_to_load + x_test_string)
        y_test = np.load(file=path_to_load + y_test_string)

    else:
        """
        Creating Reader Features
        """
        reader_features = tfr.ReaderFeatures(set_data_files=path_train_and_test_images,
                                             number_of_classes=number_of_classes,
                                             labels_set=labels_set,
                                             is_unique_csv=is_an_unique_csv, known_data_type=known_data_type,
                                             percentages_sets=percentages_sets)

        """
        Creating Reader from ReaderFeatures
        """

        tf_reader = tfr.Reader(type_problem=OPTION_PROBLEM, reader_features=reader_features,
                               settings=SETTING_OBJECT)  # Reader Object with all information

        x_train = tf_reader.x_train
        y_train = tf_reader.y_train
        x_test = tf_reader.x_test
        y_test = tf_reader.y_test

    pt("x_train", x_train.shape)
    pt("y_train", y_train.shape)
    pt("x_test", x_test.shape)
    pt("y_test", y_test.shape)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ---- END READING DATA ----
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    CMODEL = importlib.import_module(name=MODULE_NAME, package=MODELING_PACKAGE)
    CONFIG = importlib.import_module(name=MODULE_CONFIG, package=PROJECT_ID_PACKAGE)
    cmodels = CModels(setting_object=SETTING_OBJECT, option_problem=options,
                      input_data=x_train, test=x_test,
                      input_labels=y_train, test_labels=y_test,
                      number_of_classes=number_of_classes, type=None,
                      validation=None, validation_labels=None,
                      execute_background_process=True, predict_flag=IS_PREDICTION)
    CMODEL.core(cmodels, CONFIG.call())
    """
    if __name__ == '__main__':
        import multiprocessing
        multiprocessing.freeze_support()
    """

