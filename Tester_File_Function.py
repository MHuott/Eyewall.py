import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray

fp = "C:/Users/mlhuo_dkvynem/Downloads/Iota Data/20201116T160000.nc"

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
for i in range (rows):
    for j in range(columns):
        if bt.data[i][j] > 0:
            myList[i][j] = bt.data[i][j]
        else:
            myList[i][j] = 320

xMid = (math.floor)(rows/2)
yMid = (math.floor)(columns/2)

lonMid = bt.longitude.data[xMid]
latMid = bt.latitude.data[yMid]

btCenter = myList[xMid,yMid]

r1 = 10

s = myList[xMid - r1:xMid + r1,yMid - r1:yMid + r1]

rows = len(bt.longitude)
columns = len(bt.latitude)

xMid = (math.floor)(rows/2)
yMid = (math.floor)(columns/2)

lonMid = bt.longitude.data[xMid]
latMid = bt.latitude.data[yMid]

btMin = s.min()
locMin = np.where(s == s.min())
lonMin = np.mean(bt.longitude.data[xMid - r1 + locMin[0]])
latMin = np.mean(bt.latitude.data[yMid - r1 + locMin[1]])

xOffSet = xMid - r1
yOffSet = yMid - r1

p1 = myList[xOffSet + locMin[0],yOffSet + locMin[1]][0]
p2 = myList[xOffSet + locMin[0],yOffSet - locMin[1]][0]
p3 = myList[xOffSet - locMin[0],yOffSet - locMin[1]][0]
p4 = myList[xOffSet - locMin[0],yOffSet + locMin[1]][0]
p5 = myList[xOffSet + math.floor(0.707 * locMin[0][0]),yOffSet + math.floor(0.707 * locMin[1][0])]
p6 = myList[xOffSet + math.floor(0.707 * locMin[0][0]),yOffSet - math.floor(0.707 * locMin[1][0])]
p7 = myList[xOffSet - math.floor(0.707 * locMin[0][0]),yOffSet - math.floor(0.707 * locMin[1][0])]
p8 = myList[xOffSet - math.floor(0.707 * locMin[0][0]),yOffSet + math.floor(0.707 * locMin[1][0])]
pAvg = 0.125*(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)
#pAvg = 0.25*(p1 + p2 + p3 + p4)

pThresh = btMin/0.8


if pAvg > pThresh:
    indicator = 0

else:
    indicator = 1

lonP1 = bt.longitude.data[math.floor(xOffSet + locMin[0][0])]
latP1 = bt.latitude.data[math.floor(yOffSet + locMin[1][0])]
lonP2 = lonP1
latP2 = bt.latitude.data[math.floor(yOffSet - locMin[1][0])]
lonP3 = bt.longitude.data[math.floor(xOffSet - locMin[0][0])]
latP3 = latP2

print("P1 is at (" + str(lonP1) + "," + str(latP1) + ")")
print("P2 is at (" + str(lonP2) + "," + str(latP2) + ")")
print("P3 is at (" + str(lonP3) + "," + str(latP3) + ")")
