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
import src.services.modeling.CModels as models
import tensorflow as tf
import os, datetime

from src.config.Projects import Projects
from src.utils.SettingsObject import Settings
from src.utils.Prints import pt
from src.services.api.API import Petition
from src.config.GlobalSettings import GPU_TO_PREDICT

if not GPU_TO_PREDICT:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CPrediction:
    def __init__(self, current_petition: Petition, input=None):
        # Load updated config
        self.config = Projects.get_problem_config()
        # Load updated settings
        self.settings = Projects.get_settings()
        if current_petition:
            self.results = self.execute_petition_prediction(input_path=current_petition.image_src_path)
        else:
            self.results = self.execute(input=input)
        self.readable_results = self.make_readable_results(config=self.config)

    def execute_petition_prediction(self, input_path):
        # Load model
        model = self.load_model(model_fullpath=self.settings.model_path + self.config.model_name_saved)
        # Transform current image_path to an image to be predicted
        to_be_predicted = models.process_input_unity_generic(x_input=input_path,
                                                             options=self.config.options)
        # Predict image
        results = model.predict(x=to_be_predicted)
        return results

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
        result_type = 0
        if self.results[0] == 1:
            result_type = 1
        else:
            pass
        return str(result_type)

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

