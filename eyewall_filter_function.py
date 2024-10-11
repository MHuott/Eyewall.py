import numpy as np
import math
import xarray

def filter_function(bt, s, myList, r):
    rows = len(bt.longitude)
    columns = len(bt.latitude)

    xMid = (math.floor)(rows/2)
    yMid = (math.floor)(columns/2)

    lonMid = bt.longitude.data[xMid]
    latMid = bt.latitude.data[yMid]

    btMin = s.min()
    locMin = np.where(s == s.min())
    lonMin = np.mean(bt.longitude.data[xMid - r + locMin[0]])
    latMin = np.mean(bt.latitude.data[yMid - r + locMin[1]])
    
    p1 = myList[math.floor(xMid + locMin[0][0]),yMid]
    p2 = myList[math.floor(xMid - locMin[0][0]),yMid]
    p3 = myList[xMid,math.floor(yMid - locMin[1][0])]
    p4 = myList[xMid,math.floor(yMid + locMin[1][0])]
    p5 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
    p6 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
    p7 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
    p8 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
    pAvg = 0.125 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)

    pThresh = btMin/0.77


    if pAvg > pThresh:
        indicator = 0

    else:
        indicator = 1

    dis = distance * formula

    lonP1 = np.mean(bt.longitude.data[xMid + math.floor(dis * locMin[0][0])])
    latP1 = np.mean(bt.latitude.data[yMid])
    lonP2 = bt.longitude.data[xMid - math.floor(dis * locMin[0][0])]
    latP2 = bt.latitude.data[yMid]
    lonP3 = bt.longitude.data[xMid]   
    latP3 = bt.latitude.data[yMid - math.floor(dis * locMin[1][0])]
    lonP4 = bt.longitude.data[xMid]
    latP4 = bt.latitude.data[yMid + math.floor(dis * locMin[1][0])]

    lonP5 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
    latP5 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
    lonP6 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
    latP6 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]
    lonP7 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
    latP7 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
    lonP8 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
    latP8 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]

    return indicator, lonMin, latMin, p1, p2, p3, p4, p5, p6, p7, p8, pAvg, lonP1, latP1, lonP2, latP2, lonP3, latP3, lonP4, latP4, lonP5, latP5, lonP6, latP6, lonP7, latP7, lonP8, latP8
