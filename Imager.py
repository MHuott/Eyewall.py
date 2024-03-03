import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray

fp = "C:/Users/mlhuo_dkvynem/Downloads/Iota Data/20201116T164500.nc"

dataset = xarray.open_dataset(fp)
bt = dataset.mimic_tc_89GHz_bt
btMax = 0

rows = len(bt.longitude)
columns = len(bt.latitude)
x = np.linspace(bt.longitude.min().data,bt.longitude.max().data,rows)
y = np.linspace(bt.latitude.min().data, bt.latitude.max().data, columns)
X, Y = np.meshgrid(y, x)

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
        
A = [lonP1, lonP5, lonP4, lonP7, lonP2, lonP8, lonP3, lonP6, lonP1]
B = [latP1, latP5, latP4, latP7, latP2, latP8, latP3, latP6, latP1]

C = [slonP1, slonP5, slonP4, slonP7, slonP2, slonP8, slonP3, slonP6, slonP1]
D = [slatP1, slatP5, slatP4, slatP7, slatP2, slatP8, slatP3, slatP6, slatP1]

cs = plt.contourf(Y, X, bt.data, cmap="bone")
p1 = plt.plot(A, B, color = 'orange', linestyle = 'solid', linewidth = 1,
              marker = 'o', markersize = 3, markerfacecolor = 'red',
              markeredgecolor = 'red')
p2 = plt.plot(C, D, color = 'orange', linestyle = 'solid', linewidth = 1,
              marker = 'o', markersize = 3, markerfacecolor = 'green',
              markeredgecolor = 'green')
plt.colorbar()
plt.show()
