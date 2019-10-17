class Constant(object):
    """
    Constant Class
    This class contains all variable constants.
    All variables with a variable value will call this class.
    """
    # Weight first patch
    w_first_patch = 5
    # Weight second patch
    w_second_patch = 5
    # Weight number of inputs
    w_number_inputs = 1
    # First label neurons
    first_label_neurons = 32
    # Second label neurons
    second_label_neurons = 16
    # Third label neurons
    third_label_neurons = 8
    # TODO (@gabvaztor) Delete all not necessaries attributes. Document
    # attributes_to_delete: represent witch attributes set must not be save in json file when save information.
    attributes_to_delete_information = ["_input", "_test", "_input_labels", "_test_labels",
                                        "_settings_object", "_save_model_configuration",
                                        "_show_images","_show_advanced_info", "_save_model_information",
                                        "_ask_to_save_model_information", "_input_batch", "_label_batch",
                                        "_validation", "_validation_labels", "_processes", "_saves_information"]
    # attributes_to_delete: represent witch attributes set must not be save in json file when save configuration.
    attributes_to_delete_configuration = ["_input", "_test", "_input_labels", "_test_labels", "_settings_object",
                                          "_input_batch", "_label_batch", "_validation", "_validation_labels",
                                          "_processes"]