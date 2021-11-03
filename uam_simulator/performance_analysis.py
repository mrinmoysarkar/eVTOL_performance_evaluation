#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:51:40 2021

@author: mrinmoy sarkar
"""
import json 
import matplotlib.pyplot as plt
import numpy as np



class performanceanalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.agents = None
        
    def load_data(self):
        fp = open(self.filename)
        data = json.load(fp)
        self.agents = data['agents']
        return data
    
    def plot_trajectories(self):
        # plt.figure()
        for agent in self.agents:
            if '4DT_Trajectory' in agent: 
                tj = np.array(agent['4DT_Trajectory'])
                tj = self.compress_trajectory(tj)
                if tj.shape[0]>2:
                    # plt.scatter(tj[:,0],tj[:,1])
                    return tj
                    break
                
            else:
                # print(agent)
                pass
        # plt.show()
        
    def get_all_trajectories(self):
        all_tj = []
        for agent in self.agents:
            if '4DT_Trajectory' in agent: 
                tj = np.array(agent['4DT_Trajectory'])
                tj = self.compress_trajectory(tj)
                if tj.shape[0]>=2:
                    all_tj.append(tj)
        return all_tj
        
    def compress_trajectory(self, tj):
        x1 = tj[0,0]
        y1 = tj[0,1]
        x2 = tj[1,0]
        y2 = tj[1,1]
        m = (y1-y2)/(x1-x2)
        c = y1 - m*x1
        N = tj.shape[0]
        prev_speed = np.linalg.norm(tj[0,2:4])
        new_tj = []
        new_tj.append(tj[0,:])
        for i in range(1,N):
            x2 = tj[i,0]
            y2 = tj[i,1]
            cur_speed = np.linalg.norm(tj[i,2:4])
            err_speed = abs(cur_speed - prev_speed)
            err = abs(y2 - (m*x2+c))
            if err < 10e-9 and i+1 != N and err_speed < 10e-9:
                pass
            else:
                new_tj.append(tj[i,:])
                m = (y1-y2)/(x1-x2)
                c = y1 - m*x1
            prev_speed = cur_speed
            x1,y1 = x2,y2
            
        # print(new_tj)
        # print("**********************************")
        return np.array(new_tj)
           
           
           
        
        
        
if __name__ == '__main__':
    pa = performanceanalyzer('../logs/example_run.json')
    data = pa.load_data()
    pa.plot_trajectories()
    
    