import numpy as np
import math
import xarray

def hurricaneslicer(bt, myList, r):
    rows = len(bt.longitude)
    columns = len(bt.latitude)

    xMid = (math.floor)(rows / 2)
    yMid = (math.floor)(columns / 2)

    lonMid = bt.longitude.data[xMid]
    latMid = bt.latitude.data[yMid]

    p1 = myList[math.floor(xMid + r), yMid]
    p2 = myList[math.floor(xMid - r), yMid]
    p3 = myList[xMid, math.floor(yMid - r)]
    p4 = myList[xMid, math.floor(yMid + r)]
    p5 = myList[xMid + math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p6 = myList[xMid + math.floor(0.707 * r), yMid - math.floor(0.707 * r)]
    p7 = myList[xMid - math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p8 = myList[xMid - math.floor(0.707 * r), yMid - math.floor(0.707 * r)]

    pAvg = 0.125 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)

    return pAvg