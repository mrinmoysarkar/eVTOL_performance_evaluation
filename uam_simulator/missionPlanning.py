#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:36:59 2021

@author: mrinmoy sarkar
"""

import numpy as np
import matplotlib.pyplot as plt

class missionPlanning:
    def __init__(self, v_stall):
        self.v_stall = v_stall
        
    def get_missionwps(self, start, goal):
        # [x, y, z, heading, vertical_speed, horizontal_speed, time]
        # units [ft, ft, ft, radian, ft/min, ft/min, min]
        heading = np.arctan2(goal[1]-start[1], goal[0]-start[0])
        wps = []
        # wp1
        wps.append([0,0,0,heading,0,0,0])
        # wp2 vertical takeoff
        v_speed = np.random.randint(250, 501) #unit ft/min
        wps.append([0,0,50,heading,v_speed,0, 50/v_speed])
        # wp3 first climb
        h_speed = np.random.randint(10, 1.2*self.v_stall+1)*88 #unit ft/min
        d = h_speed*250/500
        wps.append([d*np.cos(heading), d*np.sin(heading), 300, heading, 500,h_speed, 250/500])
        # wp4 departure terminal procedure
        h_speed = 1.2*self.v_stall*88 #unit ft/min
        d += h_speed*2
        wps.append([d*np.cos(heading), d*np.sin(heading), 300, heading, 0, h_speed, 2])
        # wp5 second climb
        h_speed = np.random.randint(1.2*self.v_stall, 151)*88 #unit ft/min
        d += h_speed*1200/500
        wps.append([d*np.cos(heading), d*np.sin(heading), 1500, heading, 500, h_speed, 1500/500])
        #########################################################################################
        mission_range = np.linalg.norm(goal-start) #in ft
        # wp10 land
        d1 = mission_range
        v_speed = np.random.randint(0,301) #unit ft/min
        wps.append([d1*np.cos(heading), d1*np.sin(heading), 0, heading, 0, 0, 50/v_speed])
        
        # wp9 second descend
        h_speed = np.random.randint(0, 1.2*self.v_stall+1)*88 #unit ft/min
        d1 = mission_range
        wps.insert(5,[d1*np.cos(heading), d1*np.sin(heading), 50, heading, v_speed, h_speed, 250/v_speed])
        
        # wp8 arrival terminal procedure
        h_speed = 1.2*self.v_stall*88 #unit ft/min
        v_speed = np.random.randint(300, 501) #unit ft/min
        d1 = mission_range-250/v_speed
        wps.insert(5,[d1*np.cos(heading), d1*np.sin(heading), 300, heading, 0, h_speed, 2])
        
        # wp7 first descent
        h_speed = np.random.randint(1.2*self.v_stall, 151)*88 #unit ft/min
        d1 -= 1.2*self.v_stall*88*2
        wps.insert(5,[d1*np.cos(heading), d1*np.sin(heading), 300, heading, 500, h_speed, 1200/500])
        
        # wp6 cruse
        d1 -= h_speed*1200/500
        cruse_range = d1-d
        wps.insert(5,[d1*np.cos(heading), d1*np.sin(heading), 1500, heading, 0, 150*88, cruse_range/(150*88)])
        
        return np.array(wps)
        
        
if __name__ == '__main__':
    mp = missionPlanning(30)
    mission_wps = mp.get_missionwps(np.array([0,0]),np.array([100000, 100000]))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(mission_wps[:,0], mission_wps[:,1], mission_wps[:,2])

    plt.show()
    
        
        