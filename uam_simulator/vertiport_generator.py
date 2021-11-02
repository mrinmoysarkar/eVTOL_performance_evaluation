#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:57:22 2021

@author: mrinmoy sarkar
"""

import numpy as np
import pandas as pd

class vertiport:
    def __init__(self, num_vertiport, leg_distance=34725, max_capacity=10):
        self.num_vertiport = num_vertiport
        self.leg_distance = leg_distance
        self.max_capacity = max_capacity
        self.vertiport_db = None
    
    def create_vertiports(self):
        vertiport_db = np.zeros((self.num_vertiport,5)) # vertiport_id, x, y, z, capacity
        cos_x = np.cos(np.radians(60))
        sin_y = np.sin(np.radians(60))
        heagon_tempate_xy = [(-1,0),
                             (1,0),
                             (-cos_x, -sin_y), 
                             (-cos_x, sin_y), 
                             (cos_x, -sin_y), 
                             (cos_x, sin_y)]
        n = len(heagon_tempate_xy)
        vertiport_id = 1
        vertiport_db[0,:] = [vertiport_id, 0, 0, 0, np.random.randint(1,self.max_capacity+1)]
        vertiport_id += 1
        for i in range(1, self.num_vertiport):
            multiplier = self.leg_distance* ((i-1)//n + 1)
            xy = heagon_tempate_xy[(i-1)%n]
            vertiport_db[i,:] = [vertiport_id, 
                                 xy[0]*multiplier, 
                                 xy[1]*multiplier, 
                                 0, 
                                 np.random.randint(1,self.max_capacity+1)]
            vertiport_id += 1
        vertiport_db[:,1] -= np.min(vertiport_db[:,1])
        vertiport_db[:,2] -= np.min(vertiport_db[:,2])
        self.vertiport_db = vertiport_db
        
    def save_vertiports(self):
        df = pd.DataFrame(data=self.vertiport_db, columns=['vertiport_id','x','y','z','capacity'])
        df.to_csv("vertiport_db.csv", index=False)
        
        
if __name__ == '__main__':
    vp = vertiport(13)
    vp.create_vertiports()
    vp.save_vertiports()