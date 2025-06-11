'''
Author: Mitchel Huott
Name: EyewallDatasetGenerator_Main.py
Function: This is the main function where the user gets the dataset they want
          from the desired location.
'''

import numpy as np
import math
import os
import pandas

print('starting')

from EyewallDataStripper import filter

path = r'D:\2021 Season\2021_07L Grace !!'
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
    print(count)

columns = ['Primary Radius', 'Primary BT', 'Secondary Radius', 'Secondary BT']
#Save output
df = pandas.DataFrame(x, columns = columns)

df.to_excel("Genevieve Eyewall.xlsx")


print('finished')

