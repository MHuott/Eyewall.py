import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray

fp = input("Enter the .nc file you want to open: ")

dataset = xarray.open_dataset(fp)
bt = dataset.mimic_tc_89GHz_bt
btMax = 0

#print(dataset)

#print(bt)

#print(bt.longitude.min().data, bt.longitude.max().data)
#print(bt.latitude.min().data, bt.longitude.max().data)

rows = len(bt.longitude)
columns = len(bt.latitude)
x = np.linspace(bt.longitude.min().data, bt.longitude.max().data, rows)
y = np.linspace(bt.latitude.min().data, bt.latitude.max().data, columns)
X, Y = np.meshgrid(y, x)


#Converted Xarray into a numpy array
rows = len(bt.longitude)
columns = len(bt.latitude)
myList = np.zeros((rows, columns))
for i in range (rows):
    for j in range (columns):
        if bt.data[i][j] > 0:
            myList[i][j] = bt.data[i][j]
        else:
            myList[i][j] = 320

xMid = (math.floor)(rows/2)
yMid = (math.floor) (columns/2)

print(xMid)
print(yMid)

cs = plt.contourf(Y, X, bt.data, cmap="bone")
plt.colorbar()
plt.show()

radPrime = int(input("Enter an integer that defines the size of the primary eyewall grid: "))


longMid = bt.longitude.data[xMid]
latMid = bt.latitude.data[yMid]

s2 = myList[xMid - radPrime:xMid + radPrime, yMid - radPrime:yMid + radPrime]

eyewallBT = s2.min()
            
eyewallLoc = np.where(s2 == eyewallBT)

eyewallLon = bt.longitude.data[xMid + eyewallLoc[0]-radPrime]
eyewallLat = bt.latitude.data[yMid + eyewallLoc[1]-radPrime]

#print(eyewallLon)
#print(eyewallLat)
#print(eyewallLoc)

#Haversine Formula
lat1 = math.radians(latMid)
lat2 = math.radians(eyewallLat[0])
lon1 = math.radians(longMid)
lon2 = math.radians(eyewallLon[0])
dlon = lon2 - lon1
dlat = lat2 - lat1

a = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) ** 2
c = 2 * math.asin(math.sqrt(a))
r = 6371 #Radius of earth in kilometers
eyewallRadius = c * r

print("Primary eyewall is: ")
print(eyewallRadius)

radSec = math.floor(3 * np.mean(eyewallLoc[0]))

s3 = myList[xMid - radSec:xMid + radSec, yMid - radSec:yMid + radSec]

xMid2=math.floor(len(s3)/2)
yMid2=math.floor(len(s3)/2)

for i in range(2 * radSec):
    for j in range(2 * radSec):
        x1 = xMid2 - radPrime
        x2 = xMid2 + radPrime
        y1 = yMid2 - radPrime
        y2 = yMid2 + radPrime
        if i > x1 and i < x2 and j > y1 and j < y2:
            s3[i,j] = 900
cs = plt.contourf(s3)
plt.colorbar()
plt.show()

sEyewallBT = s3.min()
            
sEyewallLoc = np.where(s3 == s3.min())

loc_lon = math.floor(np.mean(sEyewallLoc[0]))
loc_lat = math.floor(np.mean(sEyewallLoc[1]))

sEyewallLon = np.mean(bt.longitude.data[xMid + loc_lon - radPrime])
sEyewallLat = np.mean(bt.latitude.data[yMid + loc_lat - radPrime])

print(sEyewallLon)
print(sEyewallLat)
print(sEyewallLoc)
plt.show()


sEyewallLatMean = np.mean(sEyewallLat)
sEyewallLonMean = np.mean(sEyewallLon)

slat1 = math.radians(latMid)
slat2 = math.radians(sEyewallLatMean)
slon1 = math.radians(longMid)
slon2 = math.radians(sEyewallLonMean)
sdlon = slon2 - slon1
sdlat = slat2 - slat1

sa = math.sin(sdlat/2) ** 2 + math.cos(slat1) * math.cos(slat2) * math.sin(sdlon/2) ** 2
sc = 2 * math.asin(math.sqrt(sa))
sr = 6371 #Radius of earth in kilometers
seyewallRadius = sc * sr
print("The Secondary Eyewall is: ")
print(seyewallRadius)
cs = plt.contourf(Y, X, bt.data, cmap="bone")
plt.colorbar()
plt.show()
    
