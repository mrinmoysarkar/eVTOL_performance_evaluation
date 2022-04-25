#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 28 1:07 AM 2022

@author: mrinmoy sarkar

"""

import pandas as pd
import os
import sys
from copy import deepcopy
import shutil

def main():
    # cluster_path = './Clustering/CLUSTERING/OUTLIER/'
    cluster_path = './Clustering/CLUSTERING/REGULAR/'

    count_less_100 = 0
    count_more_100 = 0
    sampled_dataset = None
    num_samples = 200
    for root, dirs, files in os.walk(cluster_path):
        for file in files:
            file_path  = os.path.join(root, file)
            df = pd.read_csv(file_path)
            print(df.shape)
            if df.shape[0]<=60000:
                count_less_100 += 1
            else:
                count_more_100 += 1

            df = df.sample(n=min(num_samples, df.shape[0]))
            if sampled_dataset is None:
                sampled_dataset = deepcopy(df)
            else:
                sampled_dataset = pd.concat([sampled_dataset,df], ignore_index=True)

    print('less 100: {} more 100: {}'.format(count_less_100, count_more_100))
    # return
    print(sampled_dataset.shape)
    random_sampled_df = pd.read_csv('./logs/sampled_UTM_dataset.csv')

    old = random_sampled_df['agent_id'].values
    new = sampled_dataset['agent_id'].values
    print(len(old), len(new))
    set1 = set(old)
    set2 = set(new)
    set3 = set1.intersection(set2)
    print(len(set3))

    #copy already computed samples
    src = './logs/profiles_eval/'
    dest = './logs/profiles_eval_clustering_10k_ori/'
    for agentid in set3:
        pspec = src + 'profile_spec_' + str(agentid) + '.csv'
        pfc   = src + 'profile_flight_conditions_' + str(agentid) + '.csv'
        pae   = src + 'profile_aircraft_electronics_' + str(agentid) + '.csv'
        pac   = src + 'profile_aerodynamic_coefficients_' + str(agentid) + '.csv'
        peme  = src + 'profile_electric_motor_and_propeller_efficiencies_' + str(agentid) + '.csv'
        shutil.copy(pspec, dest)
        shutil.copy(pfc, dest)
        shutil.copy(pae, dest)
        shutil.copy(pac, dest)
        shutil.copy(peme, dest)
    print("copy done")

    # Read the data from csv file and extract the configureation information
    data_part1 = pd.read_csv('./Clustering/all_UTM_sim_data_part1.csv')
    data_part2 = pd.read_csv('./Clustering/all_UTM_sim_data_part2.csv')
    data_part3 = pd.read_csv('./Clustering/all_UTM_sim_data_part3.csv')
    data_part4 = pd.read_csv('./Clustering/all_UTM_sim_data_part4.csv')
    data_part5 = pd.read_csv('./Clustering/all_UTM_sim_data_part5.csv')
    all_data = pd.concat([data_part1, data_part2, data_part3, data_part4, data_part5], ignore_index=True)

    
    lc_count = 0
    vt_count = 0
    mc_count = 0
    sampled_df_csv = None
    for agentid in sampled_dataset['agent_id']:
        df = all_data[all_data['agent_id'] == agentid]['eVTOL_type'].values
        # print(df)
        if df[0] == 'lift_and_cruse':
            lc_count += 1
        if df[0] == 'vector_thrust':
            vt_count += 1
        if df[0] == 'multicopter':
            mc_count += 1

        df1 = all_data[all_data['agent_id'] == agentid]
        sampled_df_csv = df1 if sampled_df_csv is None else pd.concat([sampled_df_csv, df1], ignore_index=True)
        

    print('lift_and_cruse', lc_count)
    print('vector_thrust', vt_count)
    print('multicopter', mc_count)

    sampled_df_csv.to_csv('./logs/sampled_UTM_dataset_from_clusters.csv', index=False)

    # lc_df = random_sampled_df[random_sampled_df['eVTOL_type']=='lift_and_cruse']
    # lc_df = lc_df.sample(n=lc_count)

    # vt_df = random_sampled_df[random_sampled_df['eVTOL_type']=='vector_thrust']
    # vt_df = vt_df.sample(n=vt_count)

    # mc_df = random_sampled_df[random_sampled_df['eVTOL_type']=='multicopter']
    # mc_df = mc_df.sample(n=mc_count)

    # new_sampled_df = pd.concat([lc_df, vt_df, mc_df], ignore_index=True)
    # src = './logs/profiles_eval/'
    # dest = './logs/profiles_eval_clustering_10k/'
    # for agentid in new_sampled_df['agent_id']:
    #     pspec = src + 'profile_spec_' + str(agentid) + '.csv'
    #     pfc   = src + 'profile_flight_conditions_' + str(agentid) + '.csv'
    #     pae   = src + 'profile_aircraft_electronics_' + str(agentid) + '.csv'
    #     pac   = src + 'profile_aerodynamic_coefficients_' + str(agentid) + '.csv'
    #     peme  = src + 'profile_electric_motor_and_propeller_efficiencies_' + str(agentid) + '.csv'
    #     shutil.copy(pspec, dest)
    #     shutil.copy(pfc, dest)
    #     shutil.copy(pae, dest)
    #     shutil.copy(pac, dest)
    #     shutil.copy(peme, dest)



if __name__ == '__main__':
    main()
