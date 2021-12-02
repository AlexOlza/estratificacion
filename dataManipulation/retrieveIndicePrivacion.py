#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:53:29 2021

@author: aolza
"""
from generarTablasVariables import load
import pandas as pd
import numpy as np
import time
import os


    



if __name__ == "__main__":
    save=True
    verbose=True
    d6,d7,d8=retrieveIndicePrivacion(save,verbose)
    