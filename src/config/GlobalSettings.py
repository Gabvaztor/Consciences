# Current problem id
PROBLEM_ID = "retinopathy_k_id"

# Set True if you are not training.
IS_PREDICTION = True

# Set True if you want to execute the API mode
API_MODE = False

# Set True if you want to execute the UI
IU_MODE = 0

# 0 = Production mode; 1 = Debug mode; 2 = Verbose mode
DEBUG_MODE = 2

# With GLOBAL_DECORATOR > 0 will wrap global decorators to all functions.
GLOBAL_DECORATOR = 1

# Set True or false if you want to use GPU
GPU_TO_TRAIN = True
GPU_TO_PREDICT = False

# When the process start this wraps all functions with two dividers: one represents the beginning of the function and
# the other one represents the end of the function.
TIMED_FLAG_DECORATOR = False

# Minimum imports. If True, it doesn't import unnecessaries libraries to execute whole project.
MINIMUM_IMPORTS = True

# Root path directory
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logger: Logger is activated by Configurator if None
LOGGER = None

LOGGER_PATH = PROJECT_ROOT_PATH + "\\logs\\"  # ...\ProjectName\src\logs\

def __modules_no_decorables():
    """
    Returns: list of all local variables that mean: paths, folders or filenames of python modules.
    This will be used in the "Configurator" to avoid them when is time of wrap decorators
    """
    init_module = "__init__.py"
    global_decorators_module = "GlobalDecorators.py"
    configurator_module = "Configurator.py"
    prints = "Prints.py"
    utils_folder = "\\utils\\"
    executor = "Executor.py"
    runner = "Runner.py"
    asynchronous_threading = "Asynchronous"

    return list(locals().copy().values())

def __functions_no_decorables():
    pt = "pt"
    return list(locals().copy().values())

MODULES_NO_DECORABLES = __modules_no_decorables()

FUNCTIONS_NO_DECORABLES = __functions_no_decorables()