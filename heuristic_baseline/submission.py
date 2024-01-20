# -*- coding: utf-8 -*-
"""heuristic_baseline.ipynb

Automatically generated.

Original file is located at:
    /home/victor/Github/splid-devkit/baseline_submissions/heuristic_python/heuristic_baseline.ipynb
"""

import pandas as pd
import numpy as np
import os
from node import Node
from datetime import datetime, timedelta
from pathlib import Path
import time

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

class index_dict:
    def __init__(self):
        self.times = self.IDADIK()
        self.indices = []
        self.AD_dex =[]
        self.modes = self.mode()

    class IDADIK:
        def __init__(self):
            self.ID = []
            self.AD = []
            self.IK = []

    class mode:
        def __init__(self):
            self.SK = []
            self.end = []

datalist = []

# Searching for training data within the dataset folder
for file in os.listdir(TEST_DATA_DIR):
    if file.endswith(".csv"):
        datalist.append(os.path.join(TEST_DATA_DIR, file))

# Sort the training data and labels
datalist = sorted(datalist)

# Print the sorted filepath to the training data
print(datalist)

frames = list()

for idx_data in range(len(datalist)):
    detected = index_dict()
    filtered = index_dict()
    
    data_path = datalist[idx_data]
    data = pd.read_csv(data_path)
    
    # Read the objectID from the filename
    filename = data_path.split('/')[-1]
    
    satcat=int(filename.split('.')[0])
    
    # Extracting longitudinal and inclination information from the pandas dataframe
    longitudes = data["Longitude (deg)"]
    inclinations = data["Inclination (deg)"]
    
    # Arbitrary assign start time and end time. Note: SNICT was developed to read in time-stamped data, 
    # however, our training data are not label with a time stamp, hence an arbitrary start and end time
    # are selected
    starttime = datetime.fromisoformat("2023-01-01 00:00:00+00:00")
    endtime = datetime.fromisoformat("2023-07-01 00:00:00+00:00")
    
    # Get std for longitude over a 24 hours window
    lon_std = []
    nodes = []
    steps_per_day = 12
    lon_was_baseline = True
    lon_baseline = 0.03
    
    for i in range(len(data["Longitude (deg)"])):
        if i <= steps_per_day:
            lon_std.append(np.std(data["Longitude (deg)"][0:steps_per_day]))
        else:
            lon_std.append(np.std(data["Longitude (deg)"][i-steps_per_day:i]))
    
    ssEW = Node(satcat=satcat,
                t=starttime,
                index=0,
                ntype="SS",
                signal="EW")
    es = Node(satcat=satcat,
                t=endtime,
                index=len(data["Longitude (deg)"])-1,
                ntype="ES",
                signal="ES",
                mtype="ES")
    
    # Run LS detection
    for i in range(steps_per_day+1,len(lon_std)-steps_per_day):             # if at least 1 day has elapsed since t0
        max_lon_std_24h = np.max(lon_std[i-steps_per_day:i])
        min_lon_std_24h = np.min(lon_std[i-steps_per_day:i])
        A = np.abs(max_lon_std_24h-min_lon_std_24h)/2
        th_ = 0.95*A
    
        # ID detection
        if (lon_std[i]>lon_baseline) & lon_was_baseline:                    # if sd is elevated & last sd was at baseline
            before = np.mean(data["Longitude (deg)"][i-steps_per_day:i])    # mean of previous day's longitudes
            after = np.mean(data["Longitude (deg)"][i:i+steps_per_day])     # mean of next day's longitudes
            # if not temporary noise, then real ID
            if np.abs(before-after)>0.3:                                    # if means are different
                lon_was_baseline = False                                    # the sd is not yet back at baseline
                index = i
                if i < steps_per_day+2:
                    ssEW.mtype = "NK"
                else:
                    detected.times.ID.append(starttime+timedelta(hours=i*2))
        # IK detection
        elif (lon_std[i]<=lon_baseline) & (not lon_was_baseline):           # elif sd is not elevated and drift has already been initialized
            drift_ended = True                                              # toggle end-of-drift boolean 
            for j in range(steps_per_day):                                  # for the next day, check...
                if np.abs(data["Longitude (deg)"][i]-data["Longitude (deg)"][i+j])>0.3:       # if the longitude changes from the current value
                    drift_ended = False                                     # the drift has not ended
            if drift_ended:                                                 # if the drift has ended
                lon_was_baseline = True                                     # the sd is back to baseline
                detected.times.IK.append(starttime+timedelta(hours=i*2))    # save tnow as end-of-drift
                detected.indices.append([index,i])                          # save indices for t-start & t-end
    
        # Last step
        elif (i == (len(lon_std)-steps_per_day-1))\
            &(not lon_was_baseline):
            detected.times.IK.append(starttime+timedelta(hours=i*2))
            detected.indices.append([index,i])
    
        # AD detection
        elif ((lon_std[i]-max_lon_std_24h>th_) or (min_lon_std_24h-lon_std[i]>th_)) & (not lon_was_baseline):          # elif sd is elevated and drift has already been initialized
            if i >= steps_per_day+3:
                detected.times.AD.append(starttime+timedelta(hours=i*2))
                detected.AD_dex.append(i)
    
    def add_node(n):
        nodes[len(nodes)-1].char_mode(
            next_index = n.index,
            lons = longitudes,
            incs = inclinations
        )
        if n.type == "AD":
            nodes[len(nodes)-1].mtype = "NK"
    
        if (nodes[len(nodes)-1].mtype != "NK"):
            filtered.indices.append([nodes[len(nodes)-1].index,n.index])
            filtered.modes.SK.append(nodes[len(nodes)-1].mtype)
            stop_NS = True if n.type == "ID" else False
            filtered.modes.end.append(stop_NS)
        nodes.append(n)
    
    toggle = True
    nodes.append(ssEW)
    if len(detected.times.IK) == 1:
        if len(detected.times.ID) == 1:
            filtered.times.ID.append(detected.times.ID[0])                                  # keep the current ID
            ID = Node(satcat,
                    detected.times.ID[0],
                    index=detected.indices[0][0],
                    ntype='ID',
                    lon=longitudes[detected.indices[0][0]],
                    signal="EW")
            add_node(ID)
        filtered.times.IK.append(detected.times.IK[0]) 
        IK = Node(satcat,
                detected.times.IK[0],
                index=detected.indices[0][1],
                ntype='IK',
                lon=longitudes[detected.indices[0][1]],
                signal="EW")
        apnd = True
        if len(detected.times.AD) == 1:
            AD = Node(satcat,
                      detected.times.AD[0],
                      index=detected.AD_dex[0],
                      ntype="AD",
                      signal="EW")
            add_node(AD)
        elif len(detected.times.AD) == 0:
            pass
        else:
            for j in range(len(detected.times.AD)):
                ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                    if j < (len(detected.times.AD)-1) else None
                if (ad.t>starttime+timedelta(hours=detected.indices[0][0]*2))&(ad.t<IK.t):
                    if apnd & (ad_next is not None):
                        if ((ad_next.t-ad.t)>timedelta(hours=24)):
                            add_node(ad)
                        else:
                            add_node(ad)
                            apnd = False
                    elif apnd & (ad_next is None):
                        add_node(ad)
                    elif (not apnd) & (ad_next is not None):
                        if ((ad_next.t-ad.t)>timedelta(hours=24)):
                            apnd = True
        if detected.indices[0][1] != (len(lon_std)-steps_per_day-1):
            add_node(IK)    
    
    for i in range(len(detected.times.IK)-1):                                 # for each longitudinal shift detection
        if toggle:                                                            # if the last ID was not discarded
            if ((starttime+timedelta(hours=detected.indices[i+1][0]*2)-detected.times.IK[i])>timedelta(hours=36)):# if the time between the current IK & next ID is longer than 48 hours
                filtered.times.ID.append(detected.times.ID[i])                # keep the current ID
                filtered.times.IK.append(detected.times.IK[i])                # keep the current IK
                ID = Node(satcat,
                        detected.times.ID[i],
                        index=detected.indices[i][0],
                        ntype='ID',
                        lon=longitudes[detected.indices[i][0]],
                        signal="EW")
                add_node(ID)
                IK = Node(satcat,
                        detected.times.IK[i],
                        index=detected.indices[i][1],
                        ntype='IK',
                        lon=longitudes[detected.indices[i][1]],
                        signal="EW")
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                    ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t>ID.t)&(ad.t<IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                    len(lon_std)-steps_per_day-1):
                    add_node(IK)    
                if i == len(detected.times.IK)-2:                             # if the next drift is the last drift
                    filtered.times.ID.append(starttime+timedelta(hours=detected.indices[i+1][0]*2))                    # keep the next ID
                    ID = Node(satcat,
                            starttime+timedelta(hours=detected.indices[i+1][0]*2),
                            index=detected.indices[i+1][0],
                            ntype='ID',
                            lon=longitudes[detected.indices[i+1][0]],
                            signal="EW")
                    add_node(ID)
                    IK = Node(satcat,
                            detected.times.IK[i+1],
                            index=detected.indices[i+1][1],
                            ntype='IK',
                            lon=longitudes[detected.indices[i+1][1]],
                            signal="EW")
                    apnd = True
                    for j in range(len(detected.times.AD)):
                        ad = Node(satcat,
                            detected.times.AD[j],
                            index=detected.AD_dex[j],
                            ntype="AD",
                            signal="EW")
                        ad_next = Node(satcat,
                            detected.times.AD[j+1],
                            index=detected.AD_dex[j+1],
                            ntype="AD",
                            signal="EW") \
                            if j < (len(detected.times.AD)-1) else None
                        if (ad.t>ID.t)&(ad.t<IK.t):
                            if apnd & (ad_next is not None):
                                if ((ad_next.t-ad.t)>timedelta(
                                    hours=24)):
                                    add_node(ad)
                                else:
                                    add_node(ad)
                                    apnd = False
                            elif apnd & (ad_next is None):
                                add_node(ad)
                            elif (not apnd) & (ad_next is not None):
                                if ((ad_next.t-ad.t)>timedelta(
                                    hours=24)):
                                    apnd = True
                    if detected.indices[i][1] != (
                        len(lon_std)-steps_per_day-1):
                        filtered.times.IK.append(detected.times.IK[i+1])      # keep the next IK
                        add_node(IK)
            else:                                                             # if the next ID and the current IK are 48 hours apart or less
                ID = Node(satcat,
                        detected.times.ID[i],
                        index=detected.indices[i][0],
                        ntype='ID',
                        lon=longitudes[detected.indices[i][0]],
                        signal="EW")                                          # keep the current ID
                add_node(ID)
                AD = Node(satcat,
                        detected.times.IK[i],
                        index=detected.indices[i][1],
                        ntype='AD',
                        lon=longitudes[detected.indices[i][1]],
                        signal="EW")                                          # change the current IK to an AD
                IK = Node(satcat,
                        detected.times.IK[i+1],
                        index=detected.indices[i+1][1],
                        ntype='IK',
                        lon=longitudes[detected.indices[i+1][1]],
                        signal="EW")                                          # exchange the current IK for the next one
                add_node(AD)
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                    ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t>ID.t)&(ad.t<IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                    len(lon_std)-steps_per_day-1):
                    add_node(IK)    
                filtered.times.ID.append(detected.times.ID[i])
                filtered.times.AD.append(detected.times.IK[i])
                filtered.times.IK.append(detected.times.IK[i+1])
                toggle = False                                                # skip the redundant drift
        else:
            toggle = True
    add_node(es)

    ssNS = Node(
            satcat=satcat,
            t=starttime,
            index=0,
            ntype="SS",
            signal="NS",
            mtype="NK")

    for j in range(len(filtered.indices)):
        indices = filtered.indices[j]
        first = True if indices[0] == 0 else False
        times = []
        dexs = []
        inc = inclinations[indices[0]:indices[1]].to_numpy()
        t = np.arange(indices[0],indices[1])*2
        rate = (steps_per_day/(indices[1]-indices[0]))*(np.max(inc)-np.min(inc))
        XIPS_inc_per_day = 0.0005 #0.035/30
        if (rate < XIPS_inc_per_day) and (indices[0] < steps_per_day) and (indices[1] > steps_per_day):
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
                
            ssNS.mtype = filtered.modes.SK[j]
        elif (rate < XIPS_inc_per_day):
            nodes.append(Node(
                satcat=satcat,
                t=times[0],
                index=dexs[0],
                ntype="IK",
                signal="NS",
                mtype=filtered.modes.SK[j]
            ))
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
        else:
            dt = [0.0]
            for i in range(len(inc)-1):
                dt.append((inc[i+1]-inc[i])/(2*60*60))
            prev = 1.0

            for i in range(len(dt)-1):
                if np.abs(dt[i])> 5.5e-7:
                    times.append(starttime+timedelta(hours=float(t[i])))
                    dexs.append(i+indices[0])
                    if (np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i]))/prev < 1.0:
                        if first and len(times)==2:
                            ssNS.mtype = filtered.modes.SK[0]
                            first = False
                    elif len(times)==2:
                        first = False
                    prev = np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i])

            if len(times)>0:
                nodes.append(Node(
                    satcat=satcat,
                    t=times[0],
                    index=dexs[0],
                    ntype="IK",
                    signal="NS",
                    mtype=filtered.modes.SK[j]
                ))
                ssNS.mtype = "NK"
                if filtered.modes.end[j]:
                    nodes.append(Node(
                        satcat=satcat,
                        t=starttime+timedelta(hours=indices[1]*2),
                        index=indices[1],
                        ntype="ID",
                        signal="NS",
                        mtype="NK"
                    ))
            elif filtered.indices[0][0] == 0:
                ssNS.mtype = filtered.modes.SK[0]
            else:
                ssNS.mtype = "NK"
    nodes.append(ssNS)
    nodes.sort(key=lambda x: x.t)
    
    # Convert timestamp back into timeindex and format the output to the correct format in a pandas dataframe
    ObjectID_list = []
    TimeIndex_list = []
    Direction_list = []
    Node_list = []
    Type_list = []
    for i in range(len(nodes)):
        ObjectID_list.append(nodes[i].satcat)
        TimeIndex_list.append(int(((nodes[i].t-starttime).days*24+(nodes[i].t-starttime).seconds/3600)/2))
        Direction_list.append(nodes[i].signal)
        Node_list.append(nodes[i].type)
        Type_list.append(nodes[i].mtype)
    
    # Initialize data of lists. 
    data = {'ObjectID': ObjectID_list, 
            'TimeIndex': TimeIndex_list,
            'Direction': Direction_list, 
            'Node': Node_list,
            'Type': Type_list} 
    
    # Create the pandas DataFrame 
    prediction_temp = pd.DataFrame(data) 
    frames.append(prediction_temp)

# Create the pandas DataFrame 
prediction = pd.concat(frames)
# print(prediction)

# Save the prediction into a csv file 
prediction.to_csv(TEST_PREDS_FP, index=False)  
print("Saved predictions to: {}".format(TEST_PREDS_FP))
time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG