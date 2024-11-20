#Eyewall Gradient

import numpy as np
import math
import xarray
import matplotlib.pyplot as plt

def eyewall_gradient(btData, radius, minLoc):
    '''
    This creates a new set of data so we can find our secondary
    eyewall location.  The first point is the location of the
    primary eyewall location.
    '''

    newLength = len(btData) - (minLoc + 1)

    new_btData = np.zeros(newLength)
    new_radius = np.zeros(newLength)

    for i in range(newLength):
        new_btData[i] = btData[i + minLoc]
        new_radius[i] = radius[i + minLoc]

    index = 0
    check = 0
    for i in range(newLength):
        if new_btData[i] < new_btData[i + 1] and check == 0:
            index = index + 1
        if new_btData[i] > new_btData[i + 1]:
            index = index + 1
            check = 1
        if new_btData[i] < new_btData[i + 1] and check == 1:
            break

    return index + minLoc