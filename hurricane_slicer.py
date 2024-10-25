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
    lonMin = bt.longitude.data[xMid - r]
    latMin = bt.longitude.data[yMid - r]

    p1 = myList[math.floor(xMid + r), yMid]
    p2 = myList[math.floor(xMid - r), yMid]
    p3 = myList[xMid, math.floor(yMid - r)]
    p4 = myList[xMid, math.floor(yMid + r)]
    p5 = myList[xMid + math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p6 = myList[xMid + math.floor(0.707 * r), yMid - math.floor(0.707 * r)]
    p7 = myList[xMid - math.floor(0.707 * r), yMid + math.floor(0.707 * r)]
    p8 = myList[xMid - math.floor(0.707 * r), yMid - math.floor(0.707 * r)]

    pAvg = 0.125 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)

    lat1 = math.radians(latMid)
    lat2 = math.radians(latMin)

    # This is the difference in longitude and latitude
    dLat = lat2 - lat1
    dLon = 0

    # This value takes the difference in the dLon and dLat and
    a = math.sin(dLat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    earthRadius = 6371  # radius of earth in kilometers

    radius = c * earthRadius


    return pAvg, radius