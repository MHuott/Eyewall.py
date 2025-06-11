'''
Author: Mitchel Huott
Name: HurricaneSlicer.py
Function: This slices the hurricane into thin disks whose BT average we find
'''

import numpy as np
import math
import xarray

def hurricane_slicer(bt, myList, r):
    rows = len(bt.longitude)
    columns = len(bt.latitude)
    xMid = (math.floor)(rows / 2)
    yMid = (math.floor)(columns / 2)
    latMid = bt.latitude.data[yMid]
    latIndex = bt.longitude.data[yMid - r]

    p1 = myList[math.floor(xMid + r), yMid]
    p2 = myList[math.floor(xMid - r), yMid]
    p3 = myList[xMid, math.floor(yMid - r)]
    p4 = myList[xMid, math.floor(yMid + r)]
    p5 = myList[xMid + math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p6 = myList[xMid + math.floor(0.707 * r), yMid - math.floor(0.707 * r)]
    p7 = myList[xMid - math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p8 = myList[xMid - math.floor(0.707 * r), yMid - math.floor(0.707 * r)]

    pAvg = 0.125 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)

    lat1 = math.radians(bt.latitude.data[yMid])
    lat2 = math.radians(bt.latitude.data[yMid + r])

    # This is the difference in longitude and latitude
    dLat = lat2 - lat1
    dLon = 0

    # This value takes the difference in the dLon and dLat and
    a = np.square(np.sin(dLat / 2))
    radius = 2 * 6371 * np.arcsin(np.sqrt(a))



    return pAvg, radius