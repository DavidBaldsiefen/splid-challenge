'''
node.py\n
Author: Liz Solera\n
Last Update: 2023-08-31 *Altered from its original configuration to analyze VCM-derived states\n
Function: SNICT Node and NodeGroup class definitions\n
Required Packages:\n
    - datetime\n
''' 

from datetime import datetime, timedelta
import numpy as np

class Node:
    def __init__(self,
                 satcat: str, 
                 t: datetime, 
                 t0: datetime = None,
                 t1: datetime = None,
                 dt : timedelta = None,
                 index : int = None,
                 next_index : int = None,
                 ntype: str = None,
                 signal: str = None,
                 lon: float = None,
                 confidence: float = None,
                 mtype: str = None):
        '''
        satcat -> 5 digit norad ID\n
        t -> timestamp\n
        t0 -> preceding epoch (optional)\n
        t1 -> proceding epoch (optional)\n
        dt -> timestep (optional)\n
        index -> index associated with timestep array (optional)\n
        ntype -> node type; options: "ID", "IK", "AD" (optional)\n
        lon -> satellite longitude (deg) at t0 (optional)
        '''
        self.satcat = int(satcat)
        self.t = t
        self.tstring = str(t.isoformat())
        if t0 is None:
            t0 = t
        if t1 is None:
            t1 = t0
        self.t0 = t0
        self.t1 = t1
        self.neep = t0
        self.dt = dt
        self.index = index
        self.next_index = None
        self.type = ntype
        self.signal = signal
        self.notes = []
        self.lon = lon
        self.confidence = confidence
        self.correlated = False
        self.time_err = 0.0
        self.mode_lons = None
        self.mode_incs = None
        self.EW_std = None
        self.NS_std = None
        self.mtype = mtype
    
    def char_mode(self,next_index=None,lons=None,incs=None):
        self.next_index = next_index
        self.mode_lons = lons[self.index:self.next_index]
        self.mode_incs = incs[self.index:self.next_index]
        EW_db = np.max(self.mode_lons) - np.min(self.mode_lons)
        EW_sd = np.std(self.mode_lons)
        EW = (EW_db-EW_sd)/EW_sd
        self.NS_std = np.std(incs)
        if self.type=="ID":
            self.mtype = "NK"
        elif self.type=="AD":
            self.mtype = "NK"
        elif self.type=="IK":
            self.mtype = "CK" if EW < 5.1 else "EK"
        elif self.type=="ES":
            self.mtype = "ES"
        elif self.type=="SS":
            self.mtype = "CK" if EW < 5.1 else "EK"

    def describe(self, ntype: str = None):
        '''Returns a description of the node type'''
        self.type = ntype if ntype!=None else self.type
        description = "unknown"
        if self.type == "ID":
            description = "Initiate Drift"
        elif self.type == "IK":
            description = "End Drift"
        elif self.type == "AD":
            description = "Adjust Drift"
        return description

    def ID(self):
        '''Returns node ID string'''
        id = (self.type+"."+str(self.satcat)+"@"+self.t.strftime("%Y-%m-%dT%H:%M"))
        # if self.dt != None:
        #     id = (self.type+"."+str(self.satcat)+"@"+self.t0.strftime("%Y-%m-%dT%H:%M")+"+"+str(int(self.dt.total_seconds()/60)))
        # else:
        #     id = (self.type+"."+str(self.satcat)+"@"+self.t0.strftime("%Y-%m-%dT%H:%M"))
        return id

    def note(self, notes: str):
        '''Creates a note for a node'''
        self.notes.append(notes)

    def see_notes(self):
        '''Prints all notes associated with a node'''
        print(self.ID())
        for i in self.notes:
            print("     ", i)
            
    def clear_notes(self):
        '''Deletes all notes associated with a node'''
        self.notes = []

    def correlate(self, t):
        self.correlated = True
        self.time_err = (t - self.t).total_seconds()/3600

class NodeGroup:
    def __init__(self, satcat: int, types, times, signals):
        self.satcat = int(satcat)
        self.types = types
        self.times = times
        self.signals = signals
        self.firsttime = times[0] if times else None
        self.lasttime = times[len(times)-1] if times else None
        self.num = len(self.types)
        icount = 0
        acount = 0
        ecount = 0
        for t in types:
            if t == 'ID':
                icount += 1
            if t == 'AD':
                acount += 1
            if t == 'IK':
                ecount += 1
        self.num_IDs = icount
        self.num_ADs = acount
        self.num_IKs = ecount
        if self.num > 1:
            self.duration = self.times[self.num-1] - self.times[0]
        else:
            self.duration = timedelta(days=5)
