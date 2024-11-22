#Eyewall Main

import numpy as np
import math
import os
import pandas

print('starting')

from eyewall_filter import filter

path = r'C:\Users\mlhuo\PycharmProjects\Eyewall.py\netcdf4Images'
#path = r'C:\Users\mlhuo\PycharmProjects\Eyewall.py'
os.chdir(path)

#Count the number of .nc files
imageCount = 0
for file in os.listdir():
    if file.endswith(".nc"):
        imageCount = imageCount + 1

#Iterate for .nc files
x = np.zeros((imageCount, 4))
count = 0
for file in os.listdir():
    if file.endswith(".nc"):
        fp = f"{path}/{file}"
        result = filter(file, count)
        x[count , :] = result
    count = count + 1

columns = ['Primary Radius', 'Primary BT', 'Secondary Radius', 'Secondary BT']
#Save output
df = pandas.DataFrame(x, columns = columns)

df.to_excel("Iota Eyewall.xlsx")


print('finished')

