#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:49:35 2021

@author: mrinmoy sarkar
"""

from queue import Queue
from uam_simulator import simulation
from uam_simulator import display
import random
import json
import time

def main():
    start_time = time.time()
    eVTOL_types = {'lift_and_cruse':49} #{'vector_thrust':74, 'lift_and_cruse':49, 'multicopter':28}
    simulation_types = ['reactive']#,'strategic']
    algo_types = {'reactive': ['MVP_Bluesky']} #, 'straight'],'strategic': ['Decoupled', 'LocalVO', 'SIPP']}
    
    for run_no in range(10):
        for eVTOL in eVTOL_types:
            for sim_type in simulation_types:
                for algo_type in algo_types[sim_type]:
                    # Simulation parameters
                    # random.seed(77)  # Set a random seed to ensure repeatability of a given run
                    simulation_name = 'run_' + sim_type + '_' + algo_type + '_' + eVTOL + '_' + str(run_no) # A string, will be used to name the log file
                    
                    minimum_separation = 300  # m, minimum distance that agents must maintain between each other, suggested by aurora expert
                    length_arena = 139000  # m, size of the square simulation area
                    max_speed = eVTOL_types[eVTOL]  # m/s, maximum velocity of the agents
                    sensing_radius = 5000
                    
                    time_step = 5
                    simulation_length = float('inf') 
                    n_agent_results = 500 
                    
                    n_intruder = 100  
                    simulation_type = sim_type 
                    algo_type = algo_type  
                    structure = None  
                    
                    sim = simulation.Simulation(length_arena,
                                                n_intruder,
                                                minimum_separation,
                                                max_speed,
                                                time_step=time_step,
                                                time_end=simulation_length,
                                                structure=structure,
                                                simulation_type=simulation_type,
                                                simulation_name=simulation_name,
                                                algorithm_type=algo_type,
                                                sensing_radius=sensing_radius,
                                                log_type='short',
                                                save_file=True,
                                                n_valid_agents_for_simulation_end=n_agent_results,
                                                stdout_to_file=False)
                    sim.run()
    end_time = time.time()
    print("Total simulation Time: {}s".format(end_time-start_time))
    
    
    
if __name__ == "__main__":
    main()