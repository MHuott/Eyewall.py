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


def filter(fp):
    print(fp)
    dataset = xarray.open_dataset(fp)
    bt = dataset.mimic_tc_89GHz_bt
    btMax = 0

    rows = len(bt.longitude)
    columns = len(bt.latitude)
    x = np.linspace(bt.longitude.min().data, bt.longitude.max().data, rows)
    y = np.linspace(bt.latitude.min().data, bt.longitude.max().data, columns)
    X, Y = np.meshgrid(y, x)
    radius = np.zeros(rows)

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

    print(xMid, yMid)

    lonMid = bt.longitude.data[xMid]
    latMid = bt.latitude.data[yMid]

    btCenter = myList[xMid, yMid]

    from hurricane_slicer import hurricaneslicer

    for r in range(yMid):
        result = hurricaneslicer(bt, myList, r)
        btData[r] = result
        radius[i] = r

    plt.style.use('ggplot')

    # make data
    x = np.linspace(0, yMid, yMid)
    y = btData

    # plot
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    plt.title("BT Avg with radius")

    plt.show()
    plt.xlabel("Not actually radius, will fix later")

    plt.savefig("output.jpg")


    return


'''
    lat1 = math.radians(latMid)
    lat2 = math.radians(latMin)
    lon1 = math.radians(lonMid)
    lon2 = math.radians(lonMin)

    #This is the difference in longitude and latitude
    dLon = lon2 - lon1
    dLat = lat2 - lat1

    #This value takes the difference in the dLon and dLat and
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 #radius of earth in kilometers

    radius_1 = c * r

    #Convert bt array into np array

    for i in range(rows):
        for j in range(columns):
            if bt.data[i][j] > 0:
                myList[i,j] = bt.data[i][j]
            else:
                myList[i,j] = 320


    #Find secondary eyewall location

    #We are defining new parameters to define the range of grid points we are searching on
    r2 = math.floor(3 * r1)

    s2 = myList[xMid - r2:xMid + r2,yMid - r2:yMid + r2]

    #This is setting up and array from the primary eyewall
    xMid2 = math.floor(len(s2) / 2)
    yMid2 = math.floor(len(s2) / 2)
    x1 = xMid2 - r1
    x2 = xMid2 + r1
    y1 = yMid2 - r1
    y2 = yMid2 + r1
'''

filter(fp)