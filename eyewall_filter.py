'''
Author: Mitchel Huott
Name: eyewall_filter.py
Function: Not really
'''

import numpy as np
import math
import xarray
import matplotlib.pyplot as plt

fp = "20201116T100000.nc"  #Break glass in case of emergency

#fp = "20201117T111500.nc" #  weird one, lets look at it later.


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

    from hurricane_slicer import hurricaneslicer

    for r in range(yMid):
        result = hurricaneslicer(bt, myList, r)
        btData[r] = result[0]
        radius[r] = result[1]

    btMin = np.min(btData)
    indexLocation = np.where(btData == btMin)
    btMinRadius = radius[indexLocation][0]

    plt.style.use('ggplot')

        # make data
    x = radius
    y = btData

        # plot
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    plt.title("BT Avg with radius")
    plt.xlabel("Radius (km)")
    plt.savefig("new" + str(count) + ".jpg")


    #plt.close('all')
    plt.show()

    return radius[indexLocation][0], btMin

#Fix matplotlib before doing any more imaging
'''
    plt.style.use('ggplot')

    # make data
    x = radius
    y = btData

    # plot
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    plt.title("BT Avg with radius")
    plt.xlabel("Radius (km)")
    plt.savefig("new.jpg")

    
'''



filter(fp, 1000)