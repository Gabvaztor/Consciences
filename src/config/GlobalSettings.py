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