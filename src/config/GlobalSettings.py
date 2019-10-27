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

# When the process start this wraps all functions with two dividers: one represents the beginning of the function and
# the other one represents the end of the function.
TIMED_FLAG_DECORATOR = False

# Logger: Logger is activated by Configurator if None
LOGGER = None

# Root path directory
import os
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def modules_no_decorables():
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

def functions_no_decorables():
    pt = "pt"
    return list(locals().copy().values())

MODULES_NO_DECORABLES = modules_no_decorables()

FUNCTIONS_NO_DECORABLES = functions_no_decorables()