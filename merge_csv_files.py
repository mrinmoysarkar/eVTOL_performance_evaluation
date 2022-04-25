#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 28 1:07 AM 2022

@author: mrinmoy sarkar

"""
import pandas as pd
import numpy as np
import os
import sys
from copy import deepcopy
import shutil
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def main():
    # profile_spec_path = './logs/profiles_eval_clustering_10k/'
    # # profile_spec_path = './logs/profiles_eval_sample/'
    # performance_metrics = ['profile_flight_conditions_', 'profile_aircraft_electronics_',
    #                        'profile_aerodynamic_coefficients_', 
    #                        'profile_electric_motor_and_propeller_efficiencies_']


    # file_spec_names = []
    # for root, dirs, files in os.walk(profile_spec_path):
    #     for file in files:
    #         if 'profile_spec' in file:
    #             file_spec_names.append(file)

    # train_files, test_files = train_test_split(file_spec_names, test_size=0.2)

    # print(len(train_files), len(test_files))

    # #create train dataset
    # outputs = [None]*len(performance_metrics)
    # inputs = None
    # for file in train_files:
    #     spec_num = file.split('_')[-1]
    #     for pidx, performance_mat in enumerate(performance_metrics):
    #         performance_path = os.path.join(root,performance_mat+spec_num)
    #         if os.path.exists(performance_path):
    #             try:
    #                 performance_df = pd.read_csv(performance_path)
    #                 outputs[pidx] = performance_df.copy() if outputs[pidx] is None \
    #                                 else pd.concat([outputs[pidx], performance_df.copy()], ignore_index=True)
    #             except:
    #                 print(performance_path)
    #         else:
    #             print(performance_path)
    #     if outputs:
    #         file_path = os.path.join(root, file)
    #         spec_df = pd.read_csv(file_path)
    #         inputs = spec_df.copy() if inputs is None \
    #                                 else pd.concat([inputs, spec_df.copy()], ignore_index=True)

    # inputs  = inputs.fillna(0)
    # outputs = [o.fillna(0) for o in outputs]
    # evtol_performance_data_dic = {'input':deepcopy(inputs), 'outputs':deepcopy(outputs)}

    # with open('train.pickle', 'wb') as handle:
    #     pickle.dump(evtol_performance_data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # with open('filename.pickle', 'rb') as handle:
    # #     b = pickle.load(handle)
        
    # #create test dataset
    # outputs = [None]*len(performance_metrics)
    # inputs = None
    # for file in test_files:
    #     spec_num = file.split('_')[-1]
    #     for pidx, performance_mat in enumerate(performance_metrics):
    #         performance_path = os.path.join(root,performance_mat+spec_num)
    #         if os.path.exists(performance_path):
    #             try:
    #                 performance_df = pd.read_csv(performance_path)
    #                 outputs[pidx] = performance_df.copy() if outputs[pidx] is None \
    #                                 else pd.concat([outputs[pidx], performance_df.copy()], ignore_index=True)
    #             except:
    #                 print(performance_path)
    #         else:
    #             print(performance_path)
    #     if outputs:
    #         file_path = os.path.join(root, file)
    #         spec_df = pd.read_csv(file_path)
    #         inputs = spec_df.copy() if inputs is None \
    #                                 else pd.concat([inputs, spec_df.copy()], ignore_index=True)

    # inputs  = inputs.fillna(0)
    # outputs = [o.fillna(0) for o in outputs]
    # test_evtol_performance_data_dic = {'input':deepcopy(inputs), 'outputs':deepcopy(outputs)}

    # with open('test.pickle', 'wb') as handle:
    #     pickle.dump(test_evtol_performance_data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(evtol_performance_data_dic['input'].shape[0], 
    #       evtol_performance_data_dic['outputs'][0].shape[0])
    # print(test_evtol_performance_data_dic['input'].shape[0], 
    #       test_evtol_performance_data_dic['outputs'][0].shape[0])

    #### train_dataX

    segment_types = ['hover_climb', 'dep_transition', 'second_climb',
       'departure_terminal_procedures', 'accel_climb', 'cruise', 'decel_descend',
       'arrival_terminal_procedure', 'second_descent', 'app_transition',
       'hover_descent']
    segment_weights = {'hover_climb':2, 'dep_transition':2, 'second_climb':2,
           'departure_terminal_procedures':2, 'accel_climb':2, 'cruise':1, 'decel_descend':2,
           'arrival_terminal_procedure':2, 'second_descent':2, 'app_transition':2,
           'hover_descent':2}
    segment_types = [[i] for i in segment_types]
    segment_type_encoder = OneHotEncoder()
    segment_type_encoder.fit(segment_types)

    def get_one_hot_encode_segment(x):
        if 'cruise' in x:
            return segment_type_encoder.transform([['cruise']]).toarray()[0]
        else:
            return segment_type_encoder.transform([[x]]).toarray()[0]
        
    print(get_one_hot_encode_segment('hover_climb'))

    evtol_types = ['lift_and_cruse', 'vector_thrust', 'multicopter']
    evtol_types = [[i] for i in evtol_types]

    evtol_type_encoder = OneHotEncoder()
    evtol_type_encoder.fit(evtol_types)

    def get_one_hot_encode_evtol(x):
        return evtol_type_encoder.transform([[x]]).toarray()[0]
        
    print(get_one_hot_encode_evtol('lift_and_cruse'))

    # train datax
    # with open('train.pickle', 'rb') as handle:
    #     evtol_performance_data_dic = pickle.load(handle)
    # data_X = None
    # x = evtol_performance_data_dic['input']
    
    # for i in range(x.shape[0]):
    #     evtol_type = evtol_performance_data_dic['outputs'][0]['eVTOL_type'][i]
    #     x_row = x.iloc[i].to_numpy()
    #     x1 = get_one_hot_encode_evtol(evtol_type)
    #     x2 = get_one_hot_encode_segment(x_row[0])
    #     x3 = x_row[1:].astype('float64')
    #     x_row = np.concatenate((x1,x2,x3))
    #     data_X = x_row if data_X is None else np.vstack((data_X,x_row))
    #     if i%1000==0:
    #         print(evtol_type)

    # with open('dataX.pickle', 'wb') as handle:
    #     pickle.dump(data_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #test dataX
    # with open('test.pickle', 'rb') as handle:
    #     test_evtol_performance_data_dic = pickle.load(handle)
    # data_X = None
    # x = test_evtol_performance_data_dic['input']
    # print(x.shape[0])
    # for i in range(x.shape[0]):
    #     evtol_type = test_evtol_performance_data_dic['outputs'][0]['eVTOL_type'][i]
    #     x_row = x.iloc[i].to_numpy()
    #     x1 = get_one_hot_encode_evtol(evtol_type)
    #     x2 = get_one_hot_encode_segment(x_row[0])
    #     x3 = x_row[1:].astype('float64')
    #     x_row = np.concatenate((x1,x2,x3))
    #     data_X = x_row if data_X is None else np.vstack((data_X,x_row))
    #     if i%1000==0:
    #         print(evtol_type)

    # with open('datatestX.pickle', 'wb') as handle:
    #     pickle.dump(data_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #train dataY
    with open('./pickle_files/train.pickle', 'rb') as handle:
        evtol_performance_data_dic = pickle.load(handle)
    performance = 'electric_motor_and_propeller_efficiencies'
    performances = ['flight_conditions', 'aircraft_electronics',
                           'aerodynamic_coefficients', 'electric_motor_and_propeller_efficiencies']
    data_Y = None
    indx = performances.index(performance)
    y = evtol_performance_data_dic['outputs'][indx]
    y.replace([np.inf, -np.inf], 0.0, inplace=True)
    if performance == 'electric_motor_and_propeller_efficiencies':
        new_cols = ['eVTOL_type', 'mission_segment', 'propeller_throttle', 'lift_throttle', 'propeller_rpm','propeller_thrust_N', 
            'propeller_torque_N_m', 'propeller_efficiency','propeller_motor_efficiency', 
            'propeller_power_coefficient','lift_rotor_rpm','lift_thrust_N','lift_torque_N_m',
            'lift_efficiency','lift_motor_efficiency','lift_power_coefficient','propeller_tip_mach',
            'lift_tip_mach','time_min', 'fesibility']
        y = y[new_cols]
    print(y.shape[0])
    for i in range(0,y.shape[0],8):
        y_row = y.iloc[i:i+8].to_numpy()
        y_row = y_row[:, 2:].astype('float64')
        y_row = np.mean(y_row, axis=0)
        data_Y = y_row if data_Y is None else np.vstack((data_Y, y_row))
        if i%10000==0:
            print('working',i)    
        
    with open('./pickle_files/dataY_electric_motor_and_propeller_efficiencies.pickle', 'wb') as handle:
        pickle.dump(data_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #test dataY
    # with open('./pickle_files/test.pickle', 'rb') as handle:
    #     test_evtol_performance_data_dic = pickle.load(handle)
    # performance = 'electric_motor_and_propeller_efficiencies'
    # performances = ['flight_conditions', 'aircraft_electronics',
    #                        'aerodynamic_coefficients', 'electric_motor_and_propeller_efficiencies']
    # data_Y = None
    # indx = performances.index(performance)
    # y = test_evtol_performance_data_dic['outputs'][indx]
    # y.replace([np.inf, -np.inf], 0.0, inplace=True)


    
    # if performance == 'electric_motor_and_propeller_efficiencies':
    #     new_cols = ['eVTOL_type', 'mission_segment', 'propeller_throttle', 'lift_throttle', 'propeller_rpm','propeller_thrust_N', 
    #         'propeller_torque_N_m', 'propeller_efficiency','propeller_motor_efficiency', 
    #         'propeller_power_coefficient','lift_rotor_rpm','lift_thrust_N','lift_torque_N_m',
    #         'lift_efficiency','lift_motor_efficiency','lift_power_coefficient','propeller_tip_mach',
    #         'lift_tip_mach','time_min', 'fesibility']
    #     y = y[new_cols]
    # print(y.shape)
    # print(y.columns)
    # print(np.sum(y['fesibility']==1), np.sum(y['fesibility']==1)/8)
    # for i in range(0,y.shape[0],8):
    #     y_row = y.iloc[i:i+8].to_numpy()
    #     y_row = y_row[:, 2:].astype('float64')
    #     y_row = np.mean(y_row, axis=0)
    #     data_Y = y_row if data_Y is None else np.vstack((data_Y, y_row))
    #     if i%10000==0:
    #         print('working',i)    
            
    # print(np.sum(data_Y[:,-1]))
        
    # with open('./pickle_files/datatestY_electric_motor_and_propeller_efficiencies.pickle', 'wb') as handle:
    #     pickle.dump(data_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    main()
