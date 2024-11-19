import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray

fp = "20201116T100000.nc"
#fp = "C:/Users/mlhuo_dkvynem/Downloads/Teddy Data/20200916T121500.nc"

dataset = xarray.open_dataset(fp)
bt = dataset.mimic_tc_89GHz_bt
btMax = 0

rows = len(bt.longitude)
columns = len(bt.latitude)
x = np.linspace(bt.longitude.min().data,bt.longitude.max().data,rows)
y = np.linspace(bt.latitude.min().data, bt.latitude.max().data, columns)
X, Y = np.meshgrid(y, x)

'''
from ImagerFilter import filter
result = filter(fp)

lonP1 = result[0]
latP1 = result[1]
lonP2 = result[2]
latP2 = result[3]
lonP3 = result[4]
latP3 = result[5]
lonP4 = result[6]
latP4 = result[7]
lonP5 = result[8]
latP5 = result[9]
lonP6 = result[10]
latP6 = result[11]
lonP7 = result[12]
latP7 = result[13]
lonP8 = result[14]
latP8 = result[15]
slonP1 = result[16]
slatP1 = result[17]
slonP2 = result[18]
slatP2 = result[19]
slonP3 = result[20]
slatP3 = result[21]
slonP4 = result[22]
slatP4 = result[23]
slonP5 = result[24]
slatP5 = result[25]
slonP6 = result[26]
slatP6 = result[27]
slonP7 = result[28]
slatP7 = result[29]
slonP8 = result[30]
slatP8 = result[31]
radius1 = result[32]
radius2 = result[33]
moat_width = result[34]
lonMin = result[35]
latMin = result[36]
        
A = [lonP1, lonP5, lonP4, lonP7, lonP2, lonP8, lonP3, lonP6, lonP1]
B = [latP1, latP5, latP4, latP7, latP2, latP8, latP3, latP6, latP1]

C = [slonP1, slonP5, slonP4, slonP7, slonP2, slonP8, slonP3, slonP6, slonP1]
D = [slatP1, slatP5, slatP4, slatP7, slatP2, slatP8, slatP3, slatP6, slatP1]

'''

cs = plt.contourf(Y, X, bt.data, cmap="bone")

#p = plt.plot(lonMin, latMin, color = 'orange', linestyle = 'none', linewidth = 1,
 #            marker = 'X', markersize = 4, markerfacecolor = 'yellowgreen',
  #           markeredgecolor = 'yellowgreen')

'''if lonP1 != 0:
    p1 = plt.plot(A, B, color = 'red', linestyle = 'solid', linewidth = 1,
                  marker = 'o', markersize = 4, markerfacecolor = 'lavender',
                  markeredgecolor = 'lavender')

if slonP1 != 0:
    p2 = plt.plot(C, D, color = 'red', linestyle = 'solid', linewidth = 2,
                  marker = 'o', markersize = 4, markerfacecolor = 'green',
                  markeredgecolor = 'green')'''


#print("Primary Radius = " + str(radius1))
#print("Secondary Radius = " + str(radius2))
#print("Moat Width = " + str(moat_width))
#plt.colorbar()

csfont = {'fontname':'Times New Roman'}

plt.title("2020/11/16 at Time:16:00:00", **csfont, fontsize = 20)
plt.xlabel("Longitude", **csfont, fontsize = 20)
plt.ylabel("Latitude", **csfont, fontsize = 20)
#plt.axis('scaled')

plt.show()

