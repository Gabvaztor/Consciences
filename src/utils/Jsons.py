import json
import traceback

import src.utils.UtilsFunctions as utils
from src.utils.Prints import pt

def object_to_json(object, attributes_to_delete=None, **kwargs):
    """
    Convert class to json with properties method.
    :param attributes_to_delete: String set with all attributes' names to delete from properties method
    :return: sort json from class properties.
    """
    try:
        object_dictionary = utils.class_properties(object=object, attributes_to_delete=attributes_to_delete)
        json_string = json.dumps(object, default=lambda m: object_dictionary, sort_keys=True, indent=4)
    except Exception as e:
        pt(e)
        pt(traceback.print_exc())
        raise ValueError("STOP")
    return json_string