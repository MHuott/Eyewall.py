#Eyewall Main

import numpy as np
import math
import os
import pandas

from EyewallScanner import eyewallScanner

#fp = "C:/Users/mlhuo_dkvynem/Downloads/Luara Data/20200826T090000.nc"
radPrime = 10

#result = eyewallScanner(fp, radPrime)

#print(result)

path = "C:/Users/mlhuo_dkvynem/Downloads/Luara Data"
os.chdir(path)

from EyewallScanner import eyewallScanner

#Count the number of .nc files
imageCount = 0
for file in os.listdir():
    if file.endswith(".nc"):
        imageCount = imageCount + 1

#Iterate for .nc files
x = np.zeros((imageCount, 10))
imageCount = 0
radPrime = 10

for file in os.listdir():
    if file.endswith(".nc"):
        fp = f"{path}/{file}"
        result = eyewallScanner(file, radPrime)
        x[imageCount,:] = result
    imageCount = imageCount + 1

#Save output
df = pandas.DataFrame(x)
df.to_excel("Eyewall.xlsx")

#fp = "C:/Users/mlhuo_dkvynem/Downloads/Luara Data/20200826T090000.nc"
#radPrime = 20

#result = eyewallScanner(fp, radPrime)

#print(x)
