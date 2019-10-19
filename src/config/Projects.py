from src.utils.SettingsObject import Settings
from .GlobalSettings import PROBLEM_ID

class Projects:
    """
    This class contains the problems id to be used.

    To use a new one, you have to add it.

    The id must be the same as the name of the folder inside "src/projects/". Example:

    problem_id = "problem_id"
    Then, in "src/projects/", you have to create a folder with the same name: "src/projects/problem_id"
    """

    # Option signals image problem
    signals_images_problem_id = 'signals_images_problems'
    # Option german prizes problem
    german_prizes_problem_id = 'german_prizes_problem'
    # Option zillow price problem
    zillow_price_problem_id = 'zillow_price_problem'
    # Option zillow price problem
    web_traffic_problem_id = 'web_traffic_problem'
    # Option Breast Cancer Wisconsin problem
    breast_cancer_wisconsin_problem_id = 'web_traffic_problem'
    # Option Retinopathy_K problem
    retinopathy_k_problem_id = "retinopathy_k_id"

    @staticmethod
    def get_settings():
        """
        Generate the path and create the Setting object from json configuration inside "src.projects" path.
        Returns: Setting object from "SETTINGS.json" of the current project id
        """
        settings_path = "projects/" + PROBLEM_ID + "/SETTINGS.json"
        print("Getting settings json from: " + settings_path)
        return Settings(settings_path)

    @staticmethod
    def get_problem_id():
        return PROBLEM_ID