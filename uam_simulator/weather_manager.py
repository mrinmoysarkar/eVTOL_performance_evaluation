#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:15:59 2021

@author: mrinmoy sarkar
"""
import pandas as pd
import numpy as np

class weather_manager:
    def __init__(self,  db_path):
        self.db_path = db_path
        self.weather_data = pd.read_csv(db_path,index_col=None)
        
    def get_wind_speed(self, location, month):
        data = self.weather_data[self.weather_data['Location']==location]
        mu = data[data['Month']==month]['speed_mean']
        std = data[data['Month']==month]['speed_std']
        speed = np.random.normal(mu,std) * 0.514444 # speed in m/s
        return speed[0]
    
    def get_wind_dirct(self, location, month):
        data = self.weather_data[self.weather_data['Location']==location]
        mu = data[data['Month']==month]['drct_mean']
        std = data[data['Month']==month]['drct_std']
        theta = np.radians(np.random.normal(mu,std)) # direction in rad
        return theta[0]
    
    def get_wind_gust_speed(self, location, month):
        data = self.weather_data[self.weather_data['Location']==location]
        mu = data[data['Month']==month]['peak_wind_gust_mean']
        std = data[data['Month']==month]['peak_wind_gust_std']
        gust_speed = np.random.normal(mu,std) * 0.514444 # speed in m/s
        return gust_speed[0]
    
    def get_wind_gust_dirct(self, location, month):
        data = self.weather_data[self.weather_data['Location']==location]
        mu = data[data['Month']==month]['peak_wind_drct_mean']
        std = data[data['Month']==month]['peak_wind_drct_std']
        gust_theta = np.radians(np.random.normal(mu,std)) # direction in rad
        return gust_theta[0]
    
    
    
if __name__ == '__main__':
    db_path = '/media/ariac/DATAPART1/mrinmoys-document/eVTOL_performance_evaluation/weather_data/Summarized_Wind_Statistics.csv'
    wm = weather_manager(db_path)
    print(wm.get_wind_speed('ATL','January'))
    print(wm.get_wind_dirct('ATL','January'))
    print(wm.get_wind_gust_speed('ATL','January'))
    print(wm.get_wind_gust_dirct('ATL','January'))