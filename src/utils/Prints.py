def pt(title=None, text=None, same_line=False):
    """
    Use the print function to print a title and an object converted to string
    Args:
        title: head comment
        text: description comment
        same_line: if True, then will do the print ending in "\r"
    """
    if isinstance(title, list) and not isinstance(title, str):
        new_title = ""
        for i, element in enumerate(title):
            if i == len(title) - 1:
                new_title += (str(element))
            else:
                new_title += (str(element) + ", ")
        title = new_title

    text_ = str(text)
    title_ = str(title)

    text = text_ if text_ != "None" else None
    title = title_ if title_ != "None" else None

    if text is None:
        text = str(title)
        title = ""
        #title = Dictionary.string_separator
    elif not title and text:
        text = str(text)
    elif title and text:
        title += ': '
    if same_line:
        print(str(title) + str(text), end="\r")
    else:
        print(str(title) + " \n " + str(text))

def show_percent_by_total(total, count_number, to_append="", same_line=False):
    progress = float(((count_number * 100) / total))
    progress = "{0:.3f}".format(progress)
    to_print = "Total Size:" + str(total) + " || " + "Count number: " + str(count_number) + " || " + \
               "Progress percent", progress + "%"
    to_print += (to_append,)
    #pt("Total Size", total, same_line=same_line)
    #pt("Count number", count_number, same_line=same_line)
    #pt("Progress percent", progress + "%", same_line=same_line)
    pt(to_print, same_line=same_line)