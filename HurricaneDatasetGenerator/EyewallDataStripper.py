'''
Author: Mitchel Huott
Name: EyewallDataStripper.py
Function: I'm not too sure
'''

import numpy as np
import math
import xarray
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

from EyewallGradient import eyewall_gradient
from scipy.signal import savgol_filter

#Break glass in case of emergency


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

    New_btData = savgol_filter(btData, 8, 4)

    primaryIndex = eyewall_gradient(New_btData, radius, 0)
    primaryBT = New_btData[primaryIndex]
    primaryRadius = radius[primaryIndex]


    secondaryIndex = eyewall_gradient(New_btData, radius, primaryIndex)


    secondaryBT = New_btData[secondaryIndex]
    secondaryRadius = radius[secondaryIndex]

    plot2 = plt.figure(figsize=(8, 6))
    plt.plot(radius, btData, linestyle='--', color='red',linewidth = 3)
    plt.plot(radius, New_btData, linewidth = 3)
    plt.title("Hurricane Sam at 1145 on 11/04", fontsize=20)
    plt.xlabel("Hurricane Radius (Km)", fontsize=18)
    plt.ylabel("Brightness Temperature", fontsize=18)
    plt.legend(['Raw Data', 'Filtered Data'], fontsize=15)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig("HurricaneSamSlice.png")
    plt.show()

    #print(primaryRadius)


    return primaryRadius, primaryBT, secondaryRadius, secondaryBT

if __name__ == '__main__':
    fp = '/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/Tropical Cyclone Data/2021 Season/2021_18L Sam/20211004T114500.nc'
    filter(fp, 1)
