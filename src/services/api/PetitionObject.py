"""
This class is used in kaggle competitions
"""

import json, os
import src.utils.Folders as folders

JSON_PETITION_NAME = "jsonPetition.json"

class Petition():

    string_folder_path = "folder"
    string_image_src_path = "imageSRC"
    string_user_ip = "userIP"
    string_model_selected = "modelSelected"
    string_date = "date"
    string_json_src_path = "jsonSRC"
    string_json_answer_src_path = "jsonAnswerSRC"

    # TODO (@gabvaztor) Create new variable that save "generic path". This means that you only have to specify only one
    # path and system will get (from a generic structure) other paths.
    def __init__(self, path=None):
        if path:
            self.path = path
            self._load_settings()
            self.__str__()
    def __str__(self):
        to_print = "\n".join([self.string_folder_path,
                              self.string_image_src_path,
                              self.string_user_ip,
                              self.string_model_selected,
                              self.string_date,
                              self.string_json_src_path,
                              self.string_json_answer_src_path,])
        return to_print

    def _load_settings(self):
        # Change with your setting file if necessary
        with open(self.path) as json_data:
            petition_object = json.load(json_data)
            self.folder_path=petition_object[self.string_folder_path]
            self.image_src_path=petition_object[self.string_image_src_path]
            self.user_ip=petition_object[self.string_user_ip]
            self.model_selected=petition_object[self.string_model_selected]
            self.date=petition_object[self.string_date]
            self.json_src_path=petition_object[self.string_json_src_path]
            self.json_answer_src_path=petition_object[self.string_json_answer_src_path]

    def load_current_configuration(self):
        """
        :return: configuration
        """
        # TODO (@gabvaztor) DOCS
        configuration = None
        folders.create_directory_from_fullpath(self.configuration_path)
        folders.create_file_from_fullpath(self.configuration_path)
        try:
            if os.stat(self.configuration_path).st_size != 0:
                with open(self.configuration_path) as json_configuration:
                    dict = json.load(json_configuration)
                    configuration = Configuration(dict)
        except Exception:
            input("Configuration problem: There is not a Configuration json file or this has nothing. Press Ok to"
                  "continue the execution and the file will be created automatically or stop the program and create for"
                  "your own")
        return configuration

class Configuration():
    def __init__(self, json_content):
        for key, value in json_content.items():
            self.__dict__[key] = value