"""
The function of this module is:
    - Getting all setting, create necessaries variables
    - Execute all necessaries tasks before main module (@decorators, etc)
    - Execute main module
"""
import sys, os

from .GlobalDecorators import DecoratorClass
from .GlobalSettings import GLOBAL_DECORATOR, DEBUG_MODE, PROJECT_ROOT_PATH

class Configurator:

    @staticmethod
    def run(modules=None):
        """
        Args:
            modules: List with modules to be executed this time
        """
        modules = list(sys.modules.values()) if not modules else modules
        print("PROJECT_SRC_PATH: ", PROJECT_ROOT_PATH)
        for module in modules.copy():
            try:
                module_path = os.path.abspath(module.__file__)
                if not PROJECT_ROOT_PATH in module_path or module_no_decorable(module_path):
                    modules.remove(module)
                else:
                    print("module_path: " + module_path)
            except Exception as e:
                pass
        if GLOBAL_DECORATOR > 0:
            print("GLOBAL_DECORATOR is: " + str(GLOBAL_DECORATOR))
            try:
                decorator_class = DecoratorClass()
                #modules.append(Configurator.__module__)
                #modules.append(decorator_class.__module__)
                decorator_class.start_wrapper_decoration(modules=modules)
                print("Global decorated finished successfully")
            except Exception as error:
                if DEBUG_MODE:
                    import traceback
                    print("Global decorated finished with errors")
                    traceback.print_exc()
                    print(str(error))
        else:
            print("GLOBAL_DECORATOR is not activated")

def module_no_decorable(module_path):
    is_not_decorable = False
    init_module = "__init__.py"
    global_decorators_module = "GlobalDecorators.py"
    configurator_module = "Configurator.py"
    modules_no_decorables_ = [init_module, global_decorators_module, configurator_module]

    for module in modules_no_decorables_:
        if module in module_path:
            is_not_decorable = True
    return is_not_decorable

