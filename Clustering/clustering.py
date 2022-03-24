# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:59:04 2022

@author: Xuyang
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.preprocessing import MinMaxScaler
import os

#--------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    output_path = 'CLUSTERING/'
    folder_outlier = 'OUTLIER/'
    folder_regular = 'REGULAR/'
    if output_path[:-1] not in os.listdir():
            os.mkdir(output_path)
    if folder_outlier[:-1] not in os.listdir(output_path):
        os.mkdir(os.path.join(output_path,folder_outlier))
    if folder_regular[:-1] not in os.listdir(output_path):
        os.mkdir(os.path.join(output_path,folder_regular))
    # Read the data from csv file and extract the configureation information
    all_data = pd.read_csv('all_UTM_sim_data.csv')
    types_evtol = np.unique(all_data['eVTOL_type'].values)
    types_sim = np.unique(all_data['simulation_type'].values)
    types_algo = np.unique(all_data['algorithm_type'].values)
    configure_info = all_data['eVTOL_type'].values+all_data['simulation_type'].values+all_data['algorithm_type'].values
    configure_info = configure_info.astype(str)
    # Compute the group combinations: 12
    groups = np.unique(configure_info)
    # A total number of 12 groups
    num_groups = len(groups)
    initial_idx = 1
    group_idx = np.ones(all_data.shape[0])
    for group in groups:
        idx = np.where(configure_info == group)[0]
        if len(idx!=0):
            group_idx[idx] = initial_idx
        initial_idx += 1
    # Extract the actual flight time, ideal flight time, and delays
    all_data['actual_flighttime'] = all_data['Actual_time_of_arrival']- all_data['Actual_time_of_departure']
    all_data['ideal_flighttime'] =  all_data['Ideal_time_of_arrival'] - all_data['Desired_time_of_departure']
    all_data['delay'] =  all_data['ideal_flighttime'] - all_data['actual_flighttime']   
    # Specify the attributes/features for clustering analysis
    selected2 = ['Number_of_conflicts', 'actual_flighttime', 'ideal_flighttime', 'delay']
    cluster_idx = np.zeros(all_data.shape[0])
    count = 0 # An offset variable to ensure the cluster idx is assigned in a consective manner
    current_sample = all_data[selected2].values
    outlier_clusters = [] # Keep track on outliers
    for i in range(len(groups)):
        # Find the indices of flight profiles in each configuration 
        current = np.where(group_idx == i+1)[0]
        # Perform the clustering analysis using DBSCAN 
        clustering = DBSCAN(eps=30, min_samples = 15).fit(current_sample[current,:])
        # Keep a record of outlier clusters 
        if -1 in np.unique(clustering.labels_):
            outlier_clusters.append(-1+count)
        # Ensure the cluster idx is assigned in a consective manner
        cluster_idx[current] = clustering.labels_ + count
        print("Number of clusters discovered in "+str(groups[i])+" : ", len(np.unique(clustering.labels_)))
        count += len(np.unique(clustering.labels_)) 
    print("-------------Clustering is finished!!!!-------------")
    print("-------------Outlier Clusters--------------")
    print(outlier_clusters)
    # Save cluster information into csv files
    cluster_idx = cluster_idx.astype(int)
    count1, count2 = 1, 1
    cluster_idxs = np.unique(cluster_idx)
    for j in range(len(cluster_idxs)):
        cluster = np.where(cluster_idx == cluster_idxs[j])[0]
        clustering_info = all_data['agent_id'].iloc[cluster]
        if cluster_idxs[j] in outlier_clusters:
            clustering_filename = 'Outlier_Cluster_' + str(count2) + '.csv'
            clustering_info.to_csv(output_path+folder_outlier+clustering_filename, index = None)
            count2 += 1
        else:
            clustering_filename = 'Regular_Cluster_' + str(count1) + '.csv'
            clustering_info.to_csv(output_path+folder_regular+clustering_filename, index = None)
            count1 += 1
    
    
    

    




    
