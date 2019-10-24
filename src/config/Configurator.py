"""
The function of this module is:
    - Getting all setting, create necessaries variables
    - Execute all necessaries tasks before main module (@decorators, etc)
    - Execute main module
"""
import sys, os

from .GlobalDecorators import DecoratorClass
from .GlobalSettings import *
from src.utils.Logger import Logger
from src.utils.Prints import pt

class Configurator:

    def __relative_imports_step_1(self):
        sys.path.append(os.path.dirname(__file__))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append('..\\..\\')
        pt("print(sys.path)",sys.path)

    def __wrap_decorators_step_2(self, modules=None):
        """
        Args:
            modules: List with modules to be executed this time.

        Returns: "_success_execution". If this must run some functionality,
        then will return True if there is not errors.
        """
        _success_execution = True
        modules = list(sys.modules.values()) if not modules else modules
        pt("PROJECT_SRC_PATH: ", PROJECT_ROOT_PATH)
        for module in modules.copy():
            try:
                module_path = os.path.abspath(module.__file__)
                #if self.__module_no_decorable(module_path) or not PROJECT_ROOT_PATH in module_path:
                if not PROJECT_ROOT_PATH in module_path:
                    modules.remove(module)
            except Exception:
                modules.remove(module)
        if GLOBAL_DECORATOR > 0:
            pt("GLOBAL_DECORATOR is: " + str(GLOBAL_DECORATOR))

            try:
                if DEBUG_MODE == 3:  # Full Verbose
                    pt("Modules to be wrapped with new functionality")
                    for i, module in enumerate(modules):
                        pt([i, module])
                decorator_class = DecoratorClass()
                #modules.append(Configurator.__module__)
                #modules.append(decorator_class.__module__)
                decorator_class.start_wrapper_decoration(modules=modules, timed_flag=TIMED_FLAG_DECORATOR,
                                                         exceptions_functions=FUNCTIONS_NO_DECORABLES)
                pt("Global decorated finished successfully")

            except Exception as error:
                if DEBUG_MODE:
                    import traceback
                    pt("Global decorated finished with errors")
                    traceback.print_exc()
                    pt(str(error))
                _success_execution = False
        else:
            pt("GLOBAL_DECORATOR is not activated")
        return _success_execution

    def __module_no_decorable(self, module_path):
        """
        Args:
            module_path: module path
        Returns: True is the "module_path" is not decorable
        """
        is_not_decorable = False
        for module in MODULES_NO_DECORABLES:
            if module in module_path:
                is_not_decorable = True
        return is_not_decorable

    def __activate_logger(self):
        global LOGGER
        LOGGER = Logger()
        return LOGGER

    def run(self, modules=None):
        self.__relative_imports_step_1()
        success = self.__wrap_decorators_step_2(modules=modules)
        self.__activate_logger()
        return success