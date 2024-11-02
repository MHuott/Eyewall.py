#Eyewall Main

import numpy as np
import math
import os
import pandas

print('starting')

from eyewall_filter import filter

path = r'C:\Users\mlhuo\PycharmProjects\Eyewall.py\netcdf4Images'
os.chdir(path)

#Count the number of .nc files
imageCount = 0
for file in os.listdir():
    if file.endswith(".nc"):
        imageCount = imageCount + 1

print(imageCount)
#Iterate for .nc files
x = np.zeros((imageCount, 2))
print(np.shape(x))
count = 0

for file in os.listdir():
    if file.endswith(".nc"):
        fp = f"{path}/{file}"
        result = filter(file, count)
        x[count , :] = result
    count = count + 1


#Save output
df = pandas.DataFrame(x)
df.to_excel("Iota Eyewall.xlsx")


print('finished')

