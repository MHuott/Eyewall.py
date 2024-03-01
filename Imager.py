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
x = np.linspace(bt.longitude.min().data,bt.longitude.max().data,rows)
y = np.linspace(bt.latitude.min().data, bt.latitude.max().data, columns)
X, Y = np.meshgrid(y, x)

cs = plt.contourf(Y, X, bt.data, cmap="bone")
plt.colorbar()
plt.show()


from ImagerFilter import filter
result = filter(fp)

radius1 = result[0]
radius2 = result[1]
lonMid = result[2]
latMid = result[3]
moat_width = result[4]
btCenter = result[5]
sLonMid = result[6]
sLatMid = result[7]
g1 = result[8]
g2 = result[9]
g3 = result[10]
g4 = result[11]

