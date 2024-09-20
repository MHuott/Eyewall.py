import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray

fp = "C:/Users/mlhuo_dkvynem/Downloads/Iota Data/20201117T053000.nc"

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

p1 = myList[math.floor(xMid + locMin[0][0]),yMid]
p2 = myList[math.floor(xMid - locMin[0][0]),yMid]
p3 = myList[xMid,math.floor(yMid - locMin[1][0])]
p4 = myList[xMid,math.floor(yMid + locMin[1][0])]
p5 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
p6 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
p7 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
p8 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
pAvg = 0.125*(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)
#pAvg = 0.25*(p1 + p2 + p3 + p4)

pThresh = btMin/0.77


if pAvg > pThresh:
    indicator = 0

else:
    indicator = 1

lonP1 = bt.longitude.data[math.floor(xMid + locMin[0][0])]
latP1 = bt.latitude.data[yMid]
lonP2 = bt.longitude.data[math.floor(xMid - locMin[0][0])]
latP2 = bt.latitude.data[yMid]
lonP3 = bt.longitude.data[xMid]
latP3 = bt.latitude.data[math.floor(yMid - locMin[1][0])]
lonP4 = bt.longitude.data[xMid]
latP4 = bt.latitude.data[math.floor(yMid + locMin[1][0])]
lonP5 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
latP5 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
lonP6 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
latP6 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]
lonP7 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
latP7 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
lonP8 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
latP8 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]

print("The mid value is (" + str(lonMid) + "," + str(latMid) + ") with P Min = ")
print("P1 is at (" + str(lonP1) + "," + str(latP1) + ") with P1 = " + str(p1))
print("P2 is at (" + str(lonP2) + "," + str(latP2) + ") with P2 = " + str(p2))
print("P3 is at (" + str(lonP3) + "," + str(latP3) + ") with P3 = " + str(p3))
print("P4 is at (" + str(lonP4) + "," + str(latP4) + ") with P4 = " + str(p4))
print("P5 is at (" + str(lonP5) + "," + str(latP5) + ") with P5 = " + str(p5))
print("P6 is at (" + str(lonP6) + "," + str(latP6) + ") with P6 = " + str(p6))
print("P7 is at (" + str(lonP7) + "," + str(latP7) + ") with P7 = " + str(p7))
print("P8 is at (" + str(lonP8) + "," + str(latP8) + ") with P8 = " + str(p8))

#r1 ordinarily changes, but I will copy and paste it to by pass this process.
#r1 = 23.253831607460686

r2 = math.floor(3 * r1) 

s2 = myList[xMid - r2:xMid + r2,yMid - r2:yMid + r2]

xMid2 = math.floor(len(s2) / 2) 
yMid2 = math.floor(len(s2) / 2)
x1 = xMid2 - r1
x2 = xMid2 + r1
y1 = yMid2 - r1
y2 = yMid2 + r1

for i in range(2 * r2):
    for j in range(2 * r2):
        if i >= x1 and i <= x2 and j >= y1 and j <= y2:
            s2[i,j] = 900

rows = len(bt.longitude)
columns = len(bt.latitude)

xMid = (math.floor)(rows/2)
yMid = (math.floor)(columns/2)

lonMid = bt.longitude.data[xMid]
latMid = bt.latitude.data[yMid]

btMin2 = s2.min()
slocMin = np.where(s2 == s2.min())
slonMin = np.mean(bt.longitude.data[xMid - r1 + locMin[0]])
slatMin = np.mean(bt.latitude.data[yMid - r1 + locMin[1]])

sp1 = myList[math.floor(xMid + locMin[0][0]),yMid]
sp2 = myList[math.floor(xMid - locMin[0][0]),yMid]
sp3 = myList[xMid,math.floor(yMid - locMin[1][0])]
sp4 = myList[xMid,math.floor(yMid + locMin[1][0])]
sp5 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
sp6 = myList[xMid + math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
sp7 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid + math.floor(0.707 * locMin[1][0])]
sp8 = myList[xMid - math.floor(0.707 * locMin[0][0]),yMid - math.floor(0.707 * locMin[1][0])]
spAvg = 0.125 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)

spThresh = btMin2/0.77


if spAvg > spThresh:
    indicator = 0

else:
    indicator = 1

slonP1 = bt.longitude.data[math.floor(xMid + locMin[0][0])]
slatP1 = bt.latitude.data[yMid]
slonP2 = bt.longitude.data[math.floor(xMid - locMin[0][0])]
slatP2 = bt.latitude.data[yMid]
slonP3 = bt.longitude.data[xMid]
slatP3 = bt.latitude.data[math.floor(yMid - locMin[1][0])]
slonP4 = bt.longitude.data[xMid]
slatP4 = bt.latitude.data[math.floor(yMid + locMin[1][0])]
slonP5 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
slatP5 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
slonP6 = bt.longitude.data[xMid + math.floor(0.707 * locMin[0][0])]
slatP6 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]
slonP7 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
slatP7 = bt.latitude.data[yMid + math.floor(0.707 * locMin[1][0])]
slonP8 = bt.longitude.data[xMid - math.floor(0.707 * locMin[0][0])]
slatP8 = bt.latitude.data[yMid - math.floor(0.707 * locMin[1][0])]

print("The mid value is (" + str(lonMid) + "," + str(latMid) + ") with P Min = ")
print("P1 is at (" + str(slonP1) + "," + str(slatP1) + ") with P1 = " + str(sp1))
print("P2 is at (" + str(slonP2) + "," + str(slatP2) + ") with P2 = " + str(sp2))
print("P3 is at (" + str(slonP3) + "," + str(slatP3) + ") with P3 = " + str(sp3))
print("P4 is at (" + str(slonP4) + "," + str(slatP4) + ") with P4 = " + str(sp4))
print("P5 is at (" + str(slonP5) + "," + str(slatP5) + ") with P5 = " + str(sp5))
print("P6 is at (" + str(slonP6) + "," + str(slatP6) + ") with P6 = " + str(sp6))
print("P7 is at (" + str(slonP7) + "," + str(slatP7) + ") with P7 = " + str(sp7))
print("P8 is at (" + str(slonP8) + "," + str(slatP8) + ") with P8 = " + str(sp8))
