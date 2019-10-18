from src.config.GlobalDecorators import DecoratorClass
import types
import sys


def das(**kwargs):
    debug = kwargs["DEBUG"] if "DEBUG" in kwargs else False
    return debug


globals = globals().copy()
current_module = globals["__file__"]

all_modules = [module for module in list(globals.values()) if module and isinstance(module, types.ModuleType)]
all_modules.append(current_module)
# for x in all_modules:
#    print(x)

if __name__ == '__main__':
    for k, v in sys.modules.items():
        print(type(k))
        print(type(v))
        print(k)
        print(v)
        print("-----------------")

    decorator_class = DecoratorClass()
    decorator_class.start_wrapper_decoration(modules=list(sys.modules.values()))

    s = das()
    print("Debug = ", str(s))