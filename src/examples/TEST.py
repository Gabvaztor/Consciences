from src.config.GlobalDecorators import DecoratorClass
import sys
def log(**kwargs):
    debug = kwargs["DEBUG"] if "DEBUG" in kwargs else False
    print(debug)
print("1515245631563126")

globals = globals().copy()

for k, v in globals.items():
    print(str(k), " ---> ", str(v))


decorator_class = DecoratorClass()
#decorator_class.start_wrapper_decoration(modules=[DecoratorClass])
modules = dir(sys.modules[__name__])
print("#~#~#~#~#~#~#~~#~#~#~#~#~#@~#@~@€~@#€@")
print("modules: ", modules)

modulenames = set(sys.modules) & set(globals)
allmodules = [sys.modules[name] for name in modulenames]
print("#~#~#~#~#~#~#~~#~#~#~#~#~#@~#@~@€~@#€@")
for x in allmodules:
    print(x)
print("#~#~#~#~#~#~#~~#~#~#~#~#~#@~#@~@€~@#€@")

current_module = globals["__file__"]
import types
all_modules = [module for module in list(globals.values()) if module and isinstance(module, types.ModuleType)]
all_modules.append(current_module)
for x in all_modules:
    print(x)

print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
for x in list(sys.modules.keys()):
    print(x)
print("X: ", current_module)
print("all modules: ", all_modules)
decorator_class.start_wrapper_decoration(modules=all_modules)

log()



