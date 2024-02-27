#Eyewall Main

import numpy as np
import math
import os
import pandas

print('starting')

from eyewall_filter import filter

#fp = "C:/Users/mlhuo_dkvynem/Downloads/Luara Data/20200826T090000.nc"

#result = eyewallScanner(fp, radPrime)

#print(result)

#path = "C:/Users/mlhuo_dkvynem/Downloads/Luara Data"
path = "C:/Users/gwilli18/Desktop/Eyewall Routine"
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
    #print(x[count, 3])

#Save output
df = pandas.DataFrame(x)
df.to_excel("Eyewall.xlsx")


print('finished')

