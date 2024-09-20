#Eyewall Main

import numpy as np
import math
import os
import pandas

print('starting')

from eyewall_filter import filter

path = "C:/Users/mlhuo_dkvynem/Downloads/Iota Data"
os.chdir(path)

#Count the number of .nc files
imageCount = 0
for file in os.listdir():
    if file.endswith(".nc"):
        imageCount = imageCount + 1

#Iterate for .nc files
x = np.zeros((imageCount, 8))
count = 0

for file in os.listdir():
    if file.endswith(".nc"):
        fp = f"{path}/{file}"
        result = filter(file)
        x[count,:] = result
    count = count + 1
    print(count)


#Save output
df = pandas.DataFrame(x)
df.to_excel("Iota Eyewall.xlsx")


print('finished')

