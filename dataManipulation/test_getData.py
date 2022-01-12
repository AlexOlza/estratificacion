#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts tests wether getData can recover the data matrix we had before (with OLDBASE)
Created on Wed Jan 12 11:19:22 2022

@author: aolza
"""
from dataManipulation.dataPreparation import getData
_,y=getData(2016,oldbase=False)
_,yOLD=getData(2016,oldbase=True)