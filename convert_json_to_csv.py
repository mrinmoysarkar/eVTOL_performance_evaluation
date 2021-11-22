# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:13:16 2021

@author: Xuyang
"""

import json
import numpy as np
import pandas as pd
import os


def conflicts_num(conflicts1, agent_id1):
    agent_counts = 0
    if len(conflicts) == 0:
        return agent_counts
    for conflict1 in conflicts1:
        agent_names = [conflict1['agent1'], conflict1['agent2']]
        if agent_id1 in agent_names:
            agent_counts += 1
    return agent_counts

#--------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    paths = ['trajectory'+str(i)+'/' for i in range(1,26)]
    num_files = [50000*(i+1) for i in range(len(paths))]
    locations = ['ATL','JFK','BOS','ORD','LAX']
    input_path = './logs/UTM_sim_data/'
    # output_path = './logs/all_trajectories/'
    output_path = './logs/'
    # if output_path[:-1] not in os.listdir():
    #     os.mkdir(output_path)
    files = os.listdir(input_path)
    vehicle_type = ['vector_thrust', 'lift_and_cruse', 'multicopter']
    agent_id = []
    eVTOL_type = []
    simul_type = []
    algo_type = []
    num_conflicts = []
    desired_time_depart = []
    actual_time_depart = []
    ideal_time_arrival = []
    actual_time_arrival = []
    preflight_time_requried = []
    average_time_collison = []
    total_planning_time = []
    weather_statistics = []
    trajectory_colnames = ['x', 'y', 'v_x', 'v_y', 't']
    row_id = 1
    file_count = 0
    for filename in files:
        if vehicle_type[0] in filename.split('.json')[0]:
            vehicle_name = vehicle_type[0]
        elif vehicle_type[1] in filename.split('.json')[0]:
            vehicle_name = vehicle_type[1]
        else:
            vehicle_name = vehicle_type[2]
        weather_stat = 'NA'   
        for loc in locations:
            if loc in filename:
                weather_stat = loc
                break
            
        # Open the jason file and read data    
        with open(input_path+filename, 'r') as myfile:
            data=myfile.read()
        obj = json.loads(data)
        agents = obj['agents']
        conflicts = obj['conflicts']
        inputs = obj['inputs']
        times = obj['times']
        file_count += 1
        print(file_count, filename)
        path_indx = 0
        for agent in agents:
            flight_status = agent['flight_status']
            varinames = agent.keys()
            if flight_status == 'finished':
                agent_id.append(row_id)
                eVTOL_type.append(vehicle_name)
                simul_type.append(inputs['simulation_type'])
                algo_type.append(inputs['algorithm_type'])
                num_conflicts.append(conflicts_num(conflicts, agent['agent_id']))
                desired_time_depart.append(agent['desired_time_of_departure'])
                actual_time_depart.append(agent['actual_time_of_departure'])
                ideal_time_arrival.append(agent['ideal_time_of_arrival'])
                actual_time_arrival.append(agent['actual_time_of_arrival'])
                if 'time_to_preflight' not in varinames:
                    preflight_time_requried.append('NA')
                else:  
                    preflight_time_requried.append(agent['time_to_preflight'])
                if 'average_time_to_plan_avoidance' not in varinames:
                    average_time_collison.append('NA')
                else:
                    average_time_collison.append(agent['average_time_to_plan_avoidance'])
                if 'total_planning_time' not in varinames:
                    total_planning_time.append('NA')
                else:
                    total_planning_time.append(agent['total_planning_time'])
                weather_statistics.append(weather_stat)
                
                # Extract the 4DT-Trajectory information into csv files
                trajectory_filename = 'Trajectory_' + str(row_id) + '.csv'
                trajectory = np.asarray(agent['4DT_Trajectory'])
                trajectory = pd.DataFrame(trajectory)
                trajectory.columns = trajectory_colnames
                trajectory.to_csv(output_path+paths[path_indx]+trajectory_filename, index = None)
                row_id += 1
                
                if row_id in num_files:
                    path_indx += 1 

                if path_indx >= len(paths):
                    path_indx = -1     
    
        myfile.close()
    # Aggregate all variables together          
    aggregate_data = [agent_id, eVTOL_type, simul_type, algo_type, num_conflicts,
                          desired_time_depart,actual_time_depart,ideal_time_arrival,
                          actual_time_arrival,preflight_time_requried, average_time_collison,
                          total_planning_time, weather_statistics]
    # Set the column names for the dataframw    
    colnames = ['agent_id','eVTOL_type','simulation_type','algorithm_type','Number_of_conflicts',
                    'Desired_time_of_departure','Actual_time_of_departure','Ideal_time_of_arrival',
                    'Actual_time_of_arrival','Time_required_for_preflight_calculation',
                    'Average_time_to_plan_avoidance', 'Total_planning_time', 
                    'Location_of_weather_statistics']
    # Save all recorded data into one dataframe  
    aggregate_info = pd.DataFrame([])  
    for i in range(len(colnames)):
        aggregate_info[colnames[i]]=aggregate_data[i]
    
    aggregate_info.to_csv('./logs/all_UTM_sim_data.csv', index = None)

        
        
        