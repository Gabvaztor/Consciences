import src.config.GlobalSettings as GS
from src.utils.SettingsObject import Settings

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
    def get_settings() -> Settings:
        """
        Generate the path and create the Setting object from json configuration inside "src.projects" path.
        Returns: Setting object from "SETTINGS.json" of the current project id
        """
        settings_path = GS.PROJECT_ROOT_PATH + "\\projects\\" + GS.PROBLEM_ID + "\\SETTINGS.json"
        try:
            settings = Settings(settings_path)
            print("Getting settings json from: " + settings_path)
        except:
            settings = Settings()
            print("Could't load Settings object from: ", settings_path)
        return settings

    @staticmethod
    def _update_project_configuration(new_project_id=None):
        """
        Update current project configuration
        """
        if new_project_id:
            GS.PROBLEM_ID = new_project_id

    @staticmethod
    def get_problem_config():
        """
        Problem_ID must be updated previously in GlobalSettings (normally in Executor step)
        Returns: Current Config class
        """
        import importlib
        MODULE_CONFIG = ".Config"
        PROJECT_ID_PACKAGE = "src.projects." + GS.PROBLEM_ID
        CONFIG = importlib.import_module(name=MODULE_CONFIG, package=PROJECT_ID_PACKAGE)
        return CONFIG.call()

    @staticmethod
    def get_problem_id():
        return GS.PROBLEM_ID