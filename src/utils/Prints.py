from src.utils.Dictionary import Dictionary

def pt(title=None, text=None, same_line=False):
    """
    Use the print function to print a title and an object coverted to string
    Args:
        title: head comment
        text: description comment
        same_line: if True, then will do the print ending in "\r"
    """
    if text is None:
        text = title
        title = Dictionary.string_separator
    else:
        title += ': '
    if same_line:
        print(str(title) + str(text), end="\r")
    else:
        print(str(title) + " \n " + str(text))