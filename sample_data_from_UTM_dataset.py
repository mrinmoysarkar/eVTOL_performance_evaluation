#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:13:15 2021

@author: mrinmoy sarkar

"""
import numpy as np
import pandas as pd
from copy import deepcopy


class sample_from_UTM_DB:
    def __init__(self, file_name):
        self.data_db = pd.read_csv(file_name)
        
    def sample_and_save(self, num_samples):
        sim_types = pd.unique(self.data_db['simulation_type'])
        eVTOL_types = pd.unique(self.data_db['eVTOL_type'])
        algo_types = pd.unique(self.data_db['algorithm_type'])
        locations = pd.unique(self.data_db['Location_of_weather_statistics'])
        mi = 10e10
        
        sampled_dataset = None
        
        for sim in sim_types:
            for eV in eVTOL_types:
                for algo in algo_types:
                    for loc in locations:
                        df = self.data_db[self.data_db['simulation_type'] == sim]
                        df = df[df['eVTOL_type'] == eV]
                        df = df[df['algorithm_type'] == algo]
                        if pd.isnull(loc):
                            df = df[pd.isnull(df['Location_of_weather_statistics'])]
                        else:
                            df = df[df['Location_of_weather_statistics'] == loc]
                        
                        if df.shape[0] != 0: 
                            if df.shape[0] < mi:
                                mi = df.shape[0]
                            print(sim, eV, algo, loc)
                            df = df.sample(n=num_samples)
                            if sampled_dataset is None:
                                sampled_dataset = deepcopy(df)
                            else:
                                sampled_dataset = pd.concat([sampled_dataset,df])
        print(mi)
        print(sampled_dataset.shape)
        sampled_dataset.to_csv('./logs/sampled_UTM_dataset.csv', index=False)
        
        
        
        
        
if __name__ == '__main__':
    num_samples = 1000
    sfud = sample_from_UTM_DB('./logs/all_UTM_sim_data.csv')
    sfud.sample_and_save(num_samples)             
        
        