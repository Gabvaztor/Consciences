"""
#3FXIAp5K
"""

from .GlobalSettings import DEBUG_MODE
from src.utils.Prints import pt

import time
import types
import functools
import inspect

class DecoratorClass(object):

    __separator = "######### "
    __end_separator = "-----------"

    def __method_separator(self, method_name, step=1):
        """
        Args:
            step: If step == 1, then is START. If step == 2, then is END with method_name. Else, END without method_name
        Returns: new str line
        """
        start = "[START] "
        end = "[END] "
        method_separator = ""
        if step == 1:
            method_separator = start + self.__separator + "\"" + method_name + "()\" " + self.__separator
        elif step == 2:
            method_separator = end + self.__separator + "\"" + method_name + "()\" " + self.__separator
        else:
            method_separator = end + self.__end_separator * 4

        return method_separator

    def __decorate_all_in_module(self, modules, function_decorator):
        for module in modules:
            for name in dir(module):
                try:
                    obj = getattr(module, name)
                    if isinstance(obj, types.FunctionType):
                        setattr(module, name, function_decorator(obj))
                    """    
                    elif isinstance(obj, type):  # Checking if is a class
                        for name, method in inspect.getmembers(obj):
                            if (not inspect.ismethod(method) and not inspect.isfunction(method))\
                                    or inspect.isbuiltin(method):
                                continue
                            print("Decorating function %s" % name)
                            setattr(obj, name, function_decorator(method))
                    """
                except Exception as e:
                    pass

    def __decorator_for_class(self, cls):
        for name, method in inspect.getmembers(cls):
            if (not inspect.ismethod(method) and not inspect.isfunction(method)) or inspect.isbuiltin(method):
                continue
            print("Decorating function %s" % name)
            setattr(cls, name, self.global_decorator(method))
        return cls

    @staticmethod
    def global_decorator(timed_flag=False):
        def msg_decorator(function):
            @functools.wraps(function)
            def inner_dec(*args, **kwargs):
                if timed_flag:
                    method_str = ""
                    start = time.time()
                    method_separator_end = ""
                    method_separator_start = ""
                    try:
                        method_str = str(function)[10:-23]
                        method_separator_start = DecoratorClass().__method_separator(method_name=method_str, step=1)
                        method_separator_end = DecoratorClass().__method_separator(method_name=method_str, step=3)
                        pt(method_separator_start)
                    except:
                        pt(method_separator_start)
                try:
                    return function(*args, DEBUG=DEBUG_MODE, **kwargs)
                except TypeError:  # Function has not kwargs
                    try:
                        return function(*args, **kwargs)
                    except:
                        return function()
                finally:
                    if timed_flag:
                        end_str = "\"" + str(method_str) + "\"" + \
                                  str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start))))
                        pt(method_separator_end + " " + end_str)
            inner_dec.__name__ = function.__name__
            inner_dec.__doc__ = function.__doc__
            return inner_dec

        return msg_decorator

    def __my_decorator(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f)
            return f(*args, **kwargs)
        return wrapper

    def start_wrapper_decoration(self, modules):
        #if not isinstance(modules, list):
        #    modules = list(modules)
        self.__decorate_all_in_module(modules=modules, function_decorator=self.global_decorator(timed_flag=False))