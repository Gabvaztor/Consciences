import json
import traceback

import src.config.GlobalSettings as GS
import src.utils.UtilsFunctions as utils
from src.utils.Prints import pt
from src.utils.Logger import Logger

LOGGER = GS.LOGGER if GS.LOGGER else Logger()

def read_json_to_dict(json_fullpath):
    """
    Read a json and return a object created from it.
    Args:
        json_fullpath: json fullpath

    Returns: json object.
    """
    try:
        with open(json_fullpath, 'r+') as outfile:
            json_readed = json.load(outfile)
        return json_readed
    except Exception as error:
        Logger().write_log_error(error)

def object_to_json(object, attributes_to_delete=None, **kwargs):
    """
    Convert class to json with properties method.
    Args:
        object: class object
        attributes_to_delete: String set with all attributes' names to delete from properties method
        **kwargs:

    Returns:sort json from class properties.

    """
    try:
        object_dictionary = utils.class_properties(object=object, attributes_to_delete=attributes_to_delete)
        json_string = json.dumps(object, default=lambda m: object_dictionary, sort_keys=True, indent=4)
    except Exception as e:
        pt(e)
        pt(traceback.print_exc())
        raise ValueError("STOP")
    return json_string

