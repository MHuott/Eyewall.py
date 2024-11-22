'''
Author: Mitchel Huott
Name: EyewallDataStripper.py
Function: I'm not too sure
'''

import numpy as np
import math
import xarray
import matplotlib.pyplot as plt
from EyewallGradient import eyewall_gradient

fp = "20201116T100000.nc"  #Break glass in case of emergency


def filter(fp, count):
    dataset = xarray.open_dataset(fp)
    bt = dataset.mimic_tc_89GHz_bt
    btMax = 0

    rows = len(bt.longitude)
    columns = len(bt.latitude)
    x = np.linspace(bt.longitude.min().data, bt.longitude.max().data, rows)
    y = np.linspace(bt.latitude.min().data, bt.longitude.max().data, columns)
    X, Y = np.meshgrid(y, x)

    #Converted Xbarray into a numpy array
    myList = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if bt.data[i][j] > 0:
                myList[i][j] = bt.data[i][j]
            else:
                myList[i][j] = 320

    xMid = (math.floor)(rows / 2)
    yMid = (math.floor)(columns / 2)

    btData = np.zeros(yMid)
    radius = np.zeros(yMid)

    lonMid = bt.longitude.data[xMid]
    latMid = bt.latitude.data[yMid]

    btCenter = myList[xMid, yMid]

    from HurricaneSlicer import hurricane_slicer

    for r in range(yMid):
        result = hurricane_slicer(bt, myList, r)
        btData[r] = result[0]
        radius[r] = result[1]

    primaryIndex = eyewall_gradient(btData, radius, 0)
    primaryBT = btData[primaryIndex]
    primaryRadius = radius[primaryIndex]


    secondaryIndex = eyewall_gradient(btData, radius, primaryIndex)


    secondaryBT = btData[secondaryIndex]
    secondaryRadius = radius[secondaryIndex]


    return primaryRadius, primaryBT, secondaryRadius, secondaryBT
