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

fp = (r"D:\2021 Season\2021_18L Sam\2021092T130000.nc")

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

    for i in range(100):
        plt.style.use('ggplot')
    csfont = {'fontname': 'Times New Roman'}

    # make data
    x = radius
    y = btData

    # plot
    fig, ax = plt.subplots()

    ax.plot(x[0:i], y[0:i], linewidth=2.0)
    plt.ylim(207, 280)
    plt.xlim(0, 500)

    plt.title("BT Avg with radius", **csfont, fontsize=20)
    plt.xlabel("Radius (km)", **csfont, fontsize=20)
    plt.ylabel("Brightness Temperature (K)", **csfont, fontsize=20)
    plt.savefig(str(i) + ".jpg")
    plt.tight_layout()
    plt.show()
    plt.close('all')

    return primaryRadius, primaryBT, secondaryRadius, secondaryBT

filter(fp, 100)

"""
for i in range(100):
    plt.style.use('ggplot')
csfont = {'fontname': 'Times New Roman'}

# make data
x = radius
y = btData

# plot
fig, ax = plt.subplots()

ax.plot(x[0:i], y[0:i], linewidth=2.0)
plt.ylim(210, 270)
plt.xlim(0, 500)

plt.title("BT Avg with radius", **csfont, fontsize=20)
plt.xlabel("Radius (km)", **csfont, fontsize=20)
plt.ylabel("Brightness Temperature (K)", **csfont, fontsize=20)
plt.savefig(str(i) + ".jpg")
plt.tight_layout()
plt.show()
plt.close('all')"""
