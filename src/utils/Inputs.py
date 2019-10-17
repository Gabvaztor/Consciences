from .Prints import pt
from .Dictionary import Dictionary

def recurrent_ask_to_save_model():
    """
    Wait user to get response to save a model
    :return:
    """
    method = recurrent_ask_to_save_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_save,
                                                   method=method)
    return response


def recurrent_ask_to_continue_without_load_model():
    """
    Wait user to get response to save a model
    :return:
    """
    method = recurrent_ask_to_continue_without_load_model
    response = recurrent_method_pass_true_or_false(question=Dictionary.string_want_to_continue_without_load,
                                                   method=method)
    return response

def recurrent_method_pass_true_or_false(question, method):
    """

    :param question:
    :param method:
    :return:
    """
    # TODO (@gabvaztor) Docs
    response = False
    pt(Dictionary.string_get_response)
    save = str(input(question + " ")).upper()
    if save == Dictionary.string_char_Y:
        response = True
    elif save != Dictionary.string_char_N:
        method()
    return response