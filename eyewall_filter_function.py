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

    xOffSet = xMid - r
    yOffSet = yMid - r
    
    p1 = myList[xOffSet + locMin[0],yOffSet + locMin[1]][0]
    p2 = myList[xOffSet + locMin[0],yOffSet - locMin[1]][0]
    p3 = myList[xOffSet - locMin[0],yOffSet - locMin[1]][0]
    p4 = myList[xOffSet - locMin[0],yOffSet + locMin[1]][0]
    pavg = 0.25*(p1+p2+p3+p4)

    pThresh = btMin/0.7


#    if p1 > pThresh or p2 > pThresh or p3 > pThresh or p4 > pThresh:
    if pavg > pThresh:
        indicator = 0

    else:
        indicator = 1
        
    return indicator, lonMin, latMin
