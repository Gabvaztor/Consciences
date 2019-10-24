class Errors(object):
    """
    Error Class
    This class contains all possibles errors.
    All Raises or exceptions will call this class.
    """

    # Validation error in percentages_set
    validation_error = "'ReaderFeatures.percentages_sets.validation' Must be lower or equal than train percentage."
    # can_not_restore_model
    can_not_restore_model = "Can not restore model. See details."
    # can_not_restore_model_because_path_not_exists
    can_not_restore_model_because_path_not_exists = "You select 'restore_model' but " \
                                                    "can not restore model because file not exists."
    # Check_dir_exists_and_create
    check_dir_exists_and_create = "Error checking file and creating folders."
    # Check the correct structure in ReaderFeatures.percentages_sets
    percentages_sets = "ReaderFeatures.percentages_sets needs to be a list with 2 or 3 positives float values " \
                       "and must sum 1."
    # Not find a label for input
    not_label_from_input = "There is not label for input"
    # Not find a label for input
    write_string_to_file = "Can't write to file. Make sure the file exists and you have privileges to write"
    # Error
    error = "Error, see description"
    # Setting Object model path bad config
    model_path_bad_configuration = "Can't save model because model path is bad configured"

    got_unexpected_parameter_debug = "got an unexpected keyword argument 'DEBUG'"