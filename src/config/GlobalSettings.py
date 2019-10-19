# Current problem id
PROBLEM_ID = "retinopathy_k_id"

# Set True if you are not training.
IS_PREDICTION = False

# 0 = Production mode; 1 = Debug mode; 2 = Verbose mode
DEBUG_MODE = 1

# With GLOBAL_DECORATOR > 0 will wrap global decorators to all functions.
GLOBAL_DECORATOR = 1

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

    return list(locals().copy().values())

MODULES_NO_DECORABLES = modules_no_decorables()