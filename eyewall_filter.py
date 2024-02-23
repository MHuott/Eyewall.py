import numpy as np
import math
import xarray

#fp = '20200822T204500.nc' Break glass in case of emergency

def filter(fp):
    dataset = xarray.open_dataset(fp)
    bt = dataset.mimic_tc_89GHz_bt
    btMax = 0

    rows = len(bt.longitude)
    columns = len(bt.latitude)
    x = np.linspace(bt.longitude.min().data, bt.longitude.max().data, rows)
    y = np.linspace(bt.latitude.min().data, bt.longitude.max().data, columns)
    X, Y = np.meshgrid(y, x)

    #Converted Xbarray into a numpy array
    myList = np.zeros((rows, columns))
    for i in range (rows):
        for j in range(columns):
            if bt.data[i][j] > 0:
                myList[i][j] = bt.data[i][j]
            else:
                myList[i][j] = 320

    xMid = (math.floor)(rows/2)
    yMid = (math.floor)(columns/2)

    lonMid = bt.longitude.data[xMid]
    latMid = bt.latitude.data[yMid]

    btCenter = myList[xMid,yMid]

    r1 = 10

    s = myList[xMid - r1:xMid + r1,yMid - r1:yMid + r1]

    ###print(r1)

 #   print(btCenter)
    
    #Remember to add the secondary eyewall values later
    if btCenter < 265 or latMid > 29:
        radius_1 = 0
        radius_2 = 0
        lonMid = 0
        latMid = 0
        sLonMid = 0
        sLatMid = 0
        moat_width = 0
        btCenter = 0
        #print('The eye has not formed')

    elif btCenter >= 265:

        from eyewall_filter_function import filter_function
        result = filter_function(bt, s, myList, r1)
#        print(result[0])
        
        pIndicator = result[0]
        lonMin = result[1]
        latMin = result[2]
   
        if pIndicator == 0:
            #print('Not a primary eyewall')
            #print('Not a secondary eyewall')
            radius_1 = 0
            radius_2 = 0
            lonMid = 0
            latMid = 0
            sLonMid = 0
            sLatMid = 0
            moat_width = 0
            btCenter = 0
        elif pIndicator != 0:

            lat1 = math.radians(latMid)
            lat2 = math.radians(latMin)
            lon1 = math.radians(lonMid)
            lon2 = math.radians(lonMin)

            #This is the difference in longitude and latitude
            dLon = lon2 - lon1
            dLat = lat2 - lat1

            #This value takes the difference in the dLon and dLat and 
            a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371 #radius of earth in kilometers

            radius_1 = c * r

            #Convert bt array into np array

            for i in range(rows):
                for j in range(columns):
                    if bt.data[i][j] > 0:
                        myList[i,j] = bt.data[i][j]                                                         
                    else:
                        myList[i,j] = 320                                                           


            #Find secondary eyewall location

            #We are defining new parameters to define the range of grid points we are searching on
            r2 = math.floor(3 * r1) 

            s2 = myList[xMid - r2:xMid + r2,yMid - r2:yMid + r2]

            #This is setting up and array from the primary eyewall
            xMid2 = math.floor(len(s2) / 2) 
            yMid2 = math.floor(len(s2) / 2)
            x1 = xMid2 - r1
            x2 = xMid2 + r1
            y1 = yMid2 - r1
            y2 = yMid2 + r1

            #This takes the array for the primary eyewall and "removes" the values by setting them super high
            for i in range(2 * r2):
                for j in range(2 * r2):
                    if i >= x1 and i <= x2 and j >= y1 and j <= y2:
                        s2[i,j] = 900

            from eyewall_filter_function import filter_function

            result = filter_function(bt, s2, myList, r1)

            sIndicator = result[0]
            sLonMid = result[1]
            sLatMid = result[2]


            if sIndicator == 0:
                #print('Not a secondary eyewall')
                radius_2 = 0
                moat_width = 0
                sLonMid = 0
                sLatMid = 0
            elif sIndicator != 0:
                #Find secondary eyewall radius using haversine formula

                lat1 = math.radians(latMid)
                lat2 = math.radians(sLatMid)
                lon1 = math.radians(lonMid)
                lon2 = math.radians(sLonMid)
    
                dLon = lon2 - lon1
                dLat = lat2 - lat1

                a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
    
                c = 2 * math.asin(math.sqrt(a))

                r = 6371 #Radius of Earth in Kilometers

                radius_2 = c * r

                moat_width = radius_2 - radius_1

    return radius_1, radius_2, lonMid, latMid, moat_width, btCenter, sLonMid, sLatMid
