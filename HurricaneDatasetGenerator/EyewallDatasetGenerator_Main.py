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

path = '/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/Tropical Cyclone Data/2021 Season/2021_12L Larry !!'
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
        print(fp)
        result = filter(file, count)
        #print(fp, result)
        x[count , :] = result
    count = count + 1

columns = ['Primary Radius', 'Primary BT', 'Secondary Radius', 'Secondary BT']
#Save output
df = pandas.DataFrame(x, columns = columns)

df.to_excel("Smoothed Genevieve Eyewall Alt.xlsx")


print('finished')

