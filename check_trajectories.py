#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday March  24 12:15:09 2022

@author: mrinmoy sarkar
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


def compress_trajectory(tj):
    # x1 = tj[0,0]
    # y1 = tj[0,1]
    # x2 = tj[1,0]
    # y2 = tj[1,1]
    # m = (y1-y2)/(x1-x2)
    # c = y1 - m*x1
    N = tj.shape[0]
    prev_speed = np.linalg.norm(tj[0,2:4])
    new_tj = []
    new_tj.append(tj[0,:])
    for i in range(1,N):
        if tj[i-1,4] > tj[i,4]:
            break
        # x2 = tj[i,0]
        # y2 = tj[i,1]
        cur_speed = np.linalg.norm(tj[i,2:4])
        err_speed = abs(cur_speed - prev_speed)
        # err = abs(y2 - (m*x2+c))
        if i+1 != N and err_speed < 1:
            pass
        else:
            new_tj.append(tj[i,:])
            # m = (y1-y2)/(x1-x2)
            # c = y1 - m*x1
        prev_speed = cur_speed
        # x1,y1 = x2,y2
        
    # print(new_tj)
    # print("**********************************")
    return np.array(new_tj)


if __name__ == '__main__':
    all_UTM_data_df = pd.read_csv('./logs/sampled_UTM_dataset.csv')
    input_path = './logs/sampled_trajectories/'
    counts = 0 
    for row_id in range(0, all_UTM_data_df.shape[0]):
        agent_id = all_UTM_data_df.iloc[row_id]['agent_id']
        trajectory_data = pd.read_csv(input_path+'Trajectory_' + str(agent_id) + '.csv')
        tj = trajectory_data.values
        tj = compress_trajectory(tj)
        if tj.shape[0]>20:
            # print(tj.shape[0])
            counts += 1
    # plt.scatter(tj[:,4], np.linalg.norm(tj[:, 2:4],axis=1))
    # # plt.scatter(tj[:,0], tj[:,1])
    # plt.show()
    print(counts)


