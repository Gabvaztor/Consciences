def __get_root_project(number_of_descent):
    import sys, os
    file = __file__
    for _ in range(number_of_descent):
        file = os.path.dirname(file)
        sys.path.append(file)

if __name__ == "__main__":
    __get_root_project(number_of_descent=4)
    from src.examples.sockets.Parameters import SERVER_PORT, CURRENT_DIR
    import subprocess
    fullpath_dir = CURRENT_DIR + r"\OpenPortBatWindows.bat"
    print("Opening:\n" + fullpath_dir + " " + str(SERVER_PORT) + "\n")
    #subprocess.call([CURRENT_DIR + r"\OpenPortBatWindows.bat " + str(SERVER_PORT)])
    p = subprocess.Popen([fullpath_dir, str(SERVER_PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()
    p.wait() # wait for process to terminate

    if not errors:
        print(output[-80:-2])
        print("Process finished successfully")

    else:
        print(errors)
        print("There was an error")