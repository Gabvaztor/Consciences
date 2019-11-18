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
import os, datetime, sys
#sys.path.append(__file__)

import src.services.modeling.CModels as models
import src.config.GlobalSettings as GS
import tensorflow as tf
from src.config.Projects import Projects
from src.utils.Prints import pt
from src.utils.PetitionObject import Petition
from src.utils.Logger import Logger

LOGGER = GS.LOGGER if GS.LOGGER else Logger()


class CPrediction():

    def __init__(self, current_petition=None, input=None, id=None):
        self.id = id if id else None
        # Load updated config
        self.config = Projects.get_problem_config()
        # Load updated settings
        self.settings = Projects.get_settings()
        if current_petition:
            self.petition_id = current_petition.petition_id if current_petition.petition_id else None
            current_petition = self.__get_petition(current_petition)
            input_path = current_petition.absolute_folder_path + current_petition.image_name
            pt("input_path", input_path)
            self.results = self.execute_petition_prediction(input_path=input_path)
        else:
            self.results = self.execute(input=input)
        self.readable_results = self.make_readable_results(config=self.config)

    def execute_petition_prediction(self, input_path):
        # Load model
        model = self.load_model(model_fullpath=self.settings.model_path + self.config.model_name_saved)

        # Transform current image_path to an image to be predicted
        to_be_predicted, _ = models.data_treatment_generic_problem(input=input_path,
                                                                   inputs_labels=None,
                                                                   options=self.config.options,
                                                                   to_predict=True)
        pt("to_be_predicted", to_be_predicted.shape)
        # Predict image
        results = model.predict(x=to_be_predicted)
        return results.tolist()

    def execute(self, input):
        # TODO (@gabvaztor) Finish
        return None

    def make_readable_results(self, config):
        readable_result = None
        if config.problem_id == Projects.retinopathy_k_problem_id:
            readable_result = self.readable_retinopathy()
        else:
            pass
        return readable_result

    def readable_retinopathy(self):
        """
        0 - No DR
        1 - Mild
        2 - Moderate
        3 - Severe
        4 - Proliferative DR
        Returns: type of retinopathy
        """
        try:
            results = str(self.results)
            max_results = str(max(self.results[0]))
            id_ = self.petition_id
            max_index = self.results[0].index(max(self.results[0]))
            labels = ["No Retinopathy", "Mild Retinopathy", "Moderate Retinopathy", "Severe Retinopathy",
                      "Proliferative Retinopathy"]

            result_type = labels[max_index]
            to_write = "RETINOPATHY" + "\nID: " + str(id_) + "\nresults: " + results + \
                       "\nmax_results: " + max_results + "\nresult_type: " + result_type

            to_show = "Most probably case: " + result_type + "<br>"
            for index, result in enumerate(self.results[0]):
                to_show += "Probability of '" + str(labels[index]) + "': " + "{0:.2f}".format(result*100) + "%<br>"
            to_show = to_show.replace("<br>","\\r\\n")
            LOGGER.write_to_logger(to_write + "\n\nTo show\n\n" + to_show)
            return to_show
        except Exception as e:
            LOGGER.write_log_error(e)

    def load_model(self, model_fullpath) -> tf.keras.Sequential:
        """
        Load and return a tf.keras model
        Args:
            model_fullpath: model fullpath

        Returns: models loaded
        """
        start_time_load_model = datetime.datetime.now()
        model = tf.keras.models.load_model(model_fullpath)
        delta = datetime.datetime.now() - start_time_load_model
        pt("Time to load model ", delta.total_seconds())
        return model

    def __get_petition(self, current_petition: Petition):
        return current_petition

"""
def previous():
    image_fullpath_to_predict = args["image_fullpath"]
    image_label_real = args["image_label_real"]
    input_data = []
    input_labels = []
    predict_flag = False
    if image_fullpath_to_predict:
        input_data = [image_fullpath_to_predict]
        predict_flag = True
    if image_label_real:
        input_labels = [image_label_real]
    train_set = [image_fullpath_to_predict]
    setting_object = Settings(Projects().get_settings())
    option_problem = Projects.get_problem_id()
    options = [option_problem, 1, 720, 1280]
    number_of_classes = 5 # Start in 0


    models = models.CModels(setting_object=setting_object, option_problem=options,
                             input_data=input_data,test=None,
                             input_labels=input_labels,test_labels=None,
                             number_of_classes=number_of_classes, type=None,
                             validation=None, validation_labels=None, predict_flag=True)
    with tf.device('/cpu:0'):  # CPU
        models.convolution_model_image()
"""