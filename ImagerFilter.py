import numpy as np
import math
import xarray

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
    if btCenter < 265:
        radius_1 = 0
        radius_2 = 0
        lonMid = 0
        latMid = 0
        sLonMid = 0
        sLatMid = 0
        moat_width = 0
        btCenter = 0
        lonP1 = 0
        latP1 = 0
        lonP2 = 0
        latP2 = 0
        lonP3 = 0
        latP3 = 0
        lonP4 = 0
        latP4 = 0
        lonP5 = 0
        latP5 = 0
        lonP6 = 0
        latP6 = 0
        lonP7 = 0
        latP7 = 0
        lonP8 = 0
        latP8 = 0
        slonP1 = 0
        slatP1 = 0
        slonP2 = 0
        slatP2 = 0
        slonP3 = 0
        slatP3 = 0
        slonP4 = 0
        slatP4 = 0
        slonP5 = 0
        slatP5 = 0
        slonP6 = 0
        slatP6 = 0
        slonP7 = 0
        slatP7 = 0
        slonP8 = 0
        slatP8 = 0
        #print('The eye has not formed')

    elif btCenter >= 265:

        from eyewall_filter_function import filter_function
        result = filter_function(bt, s, myList, r1)
#        print(result[0])
        
        pIndicator = result[0]
        lonMin = result[1]
        latMin = result[2]
        p1 = result[3]
        p2 = result[4]
        p3 = result[5]
        p4 = result[6]
        p5 = result[7]
        p6 = result[8]
        p7 = result[9]
        p8 = result[10]
        pAvg = result[11]
        lonP1 = result[12]
        latP1 = result[13]
        lonP2 = result[14]
        latP2 = result[15]
        lonP3 = result[16]
        latP3 = result[17]
        lonP4 = result[18]
        latP4 = result[19]
        lonP5 = result[20]
        latP5 = result[21]
        lonP6 = result[22]
        latP6 = result[23]
        lonP7 = result[24]
        latP7 = result[25]
        lonP8 = result[26]
        latP8 = result[27]


        
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
            lonP1 = 0
            latP1 = 0
            lonP2 = 0
            latP2 = 0
            lonP3 = 0
            latP3 = 0
            lonP4 = 0
            latP4 = 0
            lonP5 = 0
            latP5 = 0
            lonP6 = 0
            latP6 = 0
            lonP7 = 0
            latP7 = 0
            lonP8 = 0
            latP8 = 0
            slonP1 = 0
            slatP1 = 0
            slonP2 = 0
            slatP2 = 0
            slonP3 = 0
            slatP3 = 0
            slonP4 = 0
            slatP4 = 0
            slonP5 = 0
            slatP5 = 0
            slonP6 = 0
            slatP6 = 0
            slonP7 = 0
            slatP7 = 0
            slonP8 = 0
            slatP8 = 0
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
            sp1 = result[3]
            sp2 = result[4]
            sp3 = result[5]
            sp4 = result[6]
            sp5 = result[7]
            sp6 = result[8]
            sp7 = result[9]
            sp8 = result[10]
            spAvg = result[11]
            slonP1 = result[12]
            slatP1 = result[13]
            slonP2 = result[14]
            slatP2 = result[15]
            slonP3 = result[16]
            slatP3 = result[17]
            slonP4 = result[18]
            slatP4 = result[19]
            slonP5 = result[20]
            slatP5 = result[21]
            slonP6 = result[22]
            slatP6 = result[23]
            slonP7 = result[24]
            slatP7 = result[25]
            slonP8 = result[26]
            slatP8 = result[27]
        

            if sIndicator == 0:
                #print('Not a secondary eyewall')
                radius_2 = 0
                moat_width = 0
            
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

                radius_2 =   c * r

                moat_width = radius_2 - radius_1
  
    return lonP1, latP1, lonP2, latP2, lonP3, latP3, lonP4, latP4, lonP5, latP5, lonP6, latP6, lonP7, latP7, lonP8, latP8, slonP1, slatP1, slonP2, slatP2, slonP3, slatP3, slonP4, slatP4, slonP5, slatP5, slonP6, slatP6, slonP7, slatP7, slonP8, slatP8, radius_1, radius_2, moat_width, lonMin, latMin
