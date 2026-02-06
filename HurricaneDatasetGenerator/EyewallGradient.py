'''
Author: Mitchel Huott
Name: EyewallGradient.py
Function: This is a simple if statement that finds the local
minimums of the functions
'''

import numpy as np
import math
import xarray
import matplotlib.pyplot as plt

def eyewall_gradient(btData, radius, minLoc):
    new_btData = btData[minLoc + 1:]
    new_radius = radius[minLoc + 1:]

    index = 0
    check = 0
    for i in range(len(new_btData) - 1):  # Prevent out-of-bounds
        if new_btData[i] < new_btData[i + 1] and check == 0:
            index += 1
        elif new_btData[i] > new_btData[i + 1]:
            index += 1
            check = 1
        elif new_btData[i] < new_btData[i + 1] and check == 1:
            break
        if i + minLoc >= len(btData) - 2:  # More robust condition
            return -1

    return index + minLoc + 1
