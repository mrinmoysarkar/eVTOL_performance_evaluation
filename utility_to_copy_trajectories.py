#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:15:09 2021

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
import shutil

if __name__ == "__main__":
    # all_UTM_data_df = pd.read_csv('./logs/sampled_UTM_dataset.csv')
    # input_path = './logs/all_trajectories/'
    # destination = './logs/sampled_trajectories/'
    # for row_id in range(all_UTM_data_df.shape[0]):
    #     agent_id = all_UTM_data_df.iloc[row_id]['agent_id']
    #     source = input_path+'Trajectory_' + str(agent_id) + '.csv'
    #     shutil.copy(source, destination)

    #for clustering
    all_UTM_data_df = pd.read_csv('./logs/sampled_UTM_dataset_from_clusters.csv')
    input_path = '/media/ariac/DATAPART1/mrinmoys-document/eVTOL_performance_evaluation/logs/all_trajectories/'
    destination = './logs/sampled_trajectories_clustering/'
    for row_id in range(all_UTM_data_df.shape[0]):
        agent_id = all_UTM_data_df.iloc[row_id]['agent_id']
        source = input_path+'Trajectory_' + str(agent_id) + '.csv'
        shutil.copy(source, destination)