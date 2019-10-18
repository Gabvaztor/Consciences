import sys, os
from src.config.GlobalDecorators import DecoratorClass

for k, v in sys.modules.items():
    if "built-in" not in str(v) and v.__package__ is None:
        print(type(k))
        print(type(v))
        print(k)
        print(v)
        print(v.__package__)
        print("-----------------")