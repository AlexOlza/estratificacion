#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:36:16 2021

@author: alex
"""
import importlib
import joblib
import sys
import sklearn as ens
# from importlib.metadata import version
# print(version('importlib'))
# from getversion import get_module_version
def get_builtin_module_version(module ):
    # type: (...) -> str
    """
    It the module is in the list of builtin module names, the python version is returned as suggested by PEP396
    See https://www.python.org/dev/peps/pep-0396/#specification
    :param module:
    :return:
    """
    # if module.__name__ in sys.builtin_module_names:  # `imp.is_builtin` also works but maybe less efficient
        # what about this ?
        # from platform import python_version
        # python_version()
        # https://stackoverflow.com/a/25477839/7262247

        # full python version
    sys_version = '.'.join([str(v) for v in sys.version_info])
    return sys_version
    # else:
    #     raise ValueError("Module %s is not a built-in module" % module.__name__)
version = get_builtin_module_version(importlib)
print(version)

version2 = joblib.__version__
print(version2)

print(ens.show_versions())