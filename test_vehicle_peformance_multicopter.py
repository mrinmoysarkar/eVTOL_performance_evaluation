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
sys.path.append('./SUAVE_lib/trunk')
import SUAVE
from SUAVE.Core import Units , Data
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Plots.Geometry import *
import numpy as np
import pandas as pd
import time
import os
import threading

sys.path.append('./SUAVE_lib/regression/scripts/Vehicles')
sys.path.append('./uam_simulator')
# the analysis functions

# from performance_analysis import performanceanalyzer

from Electric_Multicopter import vehicle_setup




# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(start_indx, end_indx):
    global num_completed_thread
    start_time = time.time()
    all_UTM_data_df = pd.read_csv('./logs/sampled_UTM_dataset.csv')
    all_UTM_data_df = all_UTM_data_df[all_UTM_data_df['eVTOL_type']=='lift_and_cruse']
    input_path = './logs/sampled_trajectories/'
    for row_id in range(start_indx, end_indx):
        agent_id = all_UTM_data_df.iloc[row_id]['agent_id']
        eVTOL_type = all_UTM_data_df.iloc[row_id]['eVTOL_type']
        if eVTOL_type == 'lift_and_cruse':
            start_time_tj = time.time()
            trajectory_data = pd.read_csv(input_path+'Trajectory_' + str(agent_id) + '.csv')
            tj = trajectory_data.values
            tj = compress_trajectory(tj)
            
            # build the vehicle, configs, and analyses
            configs, analyses = full_setup(agent_id, tj)
            analyses.finalize()

            mission      = analyses.missions.base
            results      = mission.evaluate()

            # plot results
            plot_mission(results)
            ######
            save_results(results, profile_id=agent_id)
            # save_results_as_csv(results, agent_id)
            print(time.time()-start_time_tj, start_indx, end_indx)
    end_time = time.time()
    print("Total Analysis Time: {}s".format(end_time-start_time))
    num_completed_thread += 1




def compress_trajectory(tj):
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

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup(profile_id, tj):
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    # plot_vehicle(vehicle,plot_control_points = False)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle,tj,profile_id)
    missions_analyses = missions_setup(mission)
    
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    return configs, analyses


def base_analysis(vehicle):

    # # ------------------------------------------------------------------
    # #   Initialize the Analyses
    # # ------------------------------------------------------------------
    # analyses = SUAVE.Analyses.Vehicle()

    # # ------------------------------------------------------------------
    # #  Basic Geometry Relations
    # sizing = SUAVE.Analyses.Sizing.Sizing()
    # sizing.features.vehicle = vehicle
    # analyses.append(sizing)

    # # ------------------------------------------------------------------
    # #  Weights
    # weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    # weights.vehicle = vehicle
    # analyses.append(weights)

    # # ------------------------------------------------------------------
    # #  Aerodynamics Analysis
    # aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    # aerodynamics.geometry = vehicle
    # aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    # analyses.append(aerodynamics)

    # # ------------------------------------------------------------------
    # #  Energy
    # energy= SUAVE.Analyses.Energy.Energy()
    # energy.network = vehicle.networks
    # analyses.append(energy)


    # # # ------------------------------------------------------------------
    # # #  Noise Analysis
    # # noise = SUAVE.Analyses.Noise.Fidelity_One()
    # # noise.geometry = vehicle
    # # analyses.append(noise)

    # # ------------------------------------------------------------------
    # #  Planet Analysis
    # planet = SUAVE.Analyses.Planets.Planet()
    # analyses.append(planet)

    # # ------------------------------------------------------------------
    # #  Atmosphere Analysis
    # atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    # atmosphere.features.planet = planet.features
    # analyses.append(atmosphere)

    # return analyses

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses


# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses


# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover'
    config.networks.battery_propeller.pitch_command            = 0.  * Units.degrees
    configs.append(config)

    # ------------------------------------------------------------------
    #    Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'climb'
    config.networks.battery_propeller.pitch_command            = 0. * Units.degrees
    configs.append(config)

    return configs



def mission_setup(analyses,vehicle,tj,profile_id):
    segment_type = []
    climb_rate = []
    descend_rate = []
    start_altitude = []
    end_altitude = []
    climb_angle = []
    descent_angle = []
    speed_spec = []
    time_required = []
    
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission            = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport    = airport

    # unpack Segments module
    Segments                                                 = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    base_segment.state.numerics.number_control_points        = 8
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip

    

    # ------------------------------------------------------------------
    #   takeoff
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "hover_climb"
    segment.analyses.extend( analyses )
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 50.  * Units.ft
    segment.climb_rate                                       = max(np.random.rand(), 0.3)*500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.networks.lift_cruise.battery.max_energy
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_lift_rotor_power_coefficient=0.01,
                                                                                    initial_throttle_lift = 0.9)
    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(segment.climb_rate)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end )
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(np.nan)
    time_required.append(np.nan)

    

    # ------------------------------------------------------------------
    #   Transition
    # ------------------------------------------------------------------
    segment                                             = Segments.Climb.Constant_Speed_Constant_Angle(base_segment)
    segment.tag                                         = "dep_transition"
    segment.analyses.extend( analyses )
    segment.altitude_start                          = 50.0 * Units.ft
    segment.altitude_end                            = 300.0 * Units.ft
    segment.air_speed                               = max(np.random.rand(), 0.7) * Vstall
    segment.climb_angle                             = 6 * Units.degrees
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(segment.climb_angle)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(np.nan)

    # ------------------------------------------------------------------
    #   Departure Terminal Procedure
    # ------------------------------------------------------------------
    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag                                        = "departure_terminal_procedure"
    segment.analyses.extend( analyses )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = max(np.random.rand(), 0.5)*90.   * Units.second
    segment.air_speed                                  = max(np.random.rand(), 0.7)*Vstall
    
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment,\
                                                                                          initial_prop_power_coefficient = 0.16)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude)
    end_altitude.append(segment.altitude)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(segment.time)


    # ------------------------------------------------------------------
    #   Accelerated Climb
    # ------------------------------------------------------------------
    segment                                            = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "accel_climb"
    segment.analyses.extend( analyses )
    segment.air_speed                                  = 1.1*Vstall
    segment.altitude_start                             = 300.0 * Units.ft
    segment.altitude_end                               = 1500. * Units.ft
    segment.climb_rate                                 = 500. * Units['ft/min']
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(segment.climb_rate)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(np.nan)
    

    
    total_t = 0
    speed = 0
    for i in range(tj.shape[0]-1):
        total_t = tj[i+1,4] - tj[i,4]
        speed = np.linalg.norm(tj[i,2:4])
        # ------------------------------------------------------------------
        #   Cruise
        # ------------------------------------------------------------------
        segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
        segment.tag                                        = "cruise_"+str(i)
        segment.analyses.extend( analyses )
        segment.altitude                                   = 1500.0 * Units.ft
        segment.time                                       = total_t * Units['s'] #600.   * Units.second
        segment.air_speed                                  = max(speed, 1.1*Vstall) * Units['m/s']      #1.2*Vstall
        # segment.state.unknowns.throttle =  0.80 * ones_row(1)
        segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment,\
                                                                                              initial_prop_power_coefficient = 0.16)
    
        # add to misison
        mission.append_segment(segment)
        
        segment_type.append(segment.tag)
        climb_rate.append(np.nan)
        descend_rate.append(np.nan)
        start_altitude.append(segment.altitude)
        end_altitude.append(segment.altitude)
        climb_angle.append(np.nan)
        descent_angle.append(np.nan)
        speed_spec.append(segment.air_speed)
        time_required.append(segment.time)
    
    
    # ------------------------------------------------------------------
    #   Decelerated Descend
    # ------------------------------------------------------------------
    segment                                            = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "decel_descend"
    segment.analyses.extend( analyses )
    
    segment.altitude_start                             = 1500 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.descent_rate                               = 500. * Units['ft/min']
    segment.air_speed                                  = 1.1*Vstall
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(segment.descent_rate)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(np.nan)
    
    # ------------------------------------------------------------------
    #   Arrival Terminal Procedure
    # ------------------------------------------------------------------
    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag                                        = "arrival_terminal_procedure"
    segment.analyses.extend( analyses )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = max(np.random.rand(), 0.5)*90.   * Units.second
    segment.air_speed                                  = max(np.random.rand(), 0.7)*Vstall
    
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment,\
                                                                                          initial_prop_power_coefficient = 0.16)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude)
    end_altitude.append(segment.altitude)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(segment.time)
    
    # ------------------------------------------------------------------
    #   Transition
    # ------------------------------------------------------------------
    segment                                             = Segments.Descent.Constant_Speed_Constant_Angle(base_segment)
    segment.tag                                         = "app_transition"
    segment.analyses.extend( analyses )
    segment.altitude_start                          = 300.0 * Units.ft
    segment.altitude_end                            = 50.0 * Units.ft
    segment.air_speed                               = max(np.random.rand(), 0.65) * Vstall
    segment.descent_angle                           = 6 * Units.degrees
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(np.nan)
    descent_angle.append(segment.descent_angle)
    speed_spec.append(segment.air_speed)
    time_required.append(np.nan)
    
    # ------------------------------------------------------------------
    #   Land
    # ------------------------------------------------------------------
    segment                                            = Segments.Hover.Descent(base_segment)  
    segment.tag                                        = "hover_descent"
    segment.analyses.extend( analyses )
    
    segment.altitude_start            = 50.    * Units.ft    
    segment.altitude_end              = 0.0  * Units.ft  
    segment.descent_rate   = max(np.random.rand(), 0.5)*300.  * Units['ft/min']
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(np.nan)
    descend_rate.append(segment.descent_rate)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(np.nan)
    time_required.append(np.nan)
    
    profile_spec_df = pd.DataFrame(data={'segment_type':segment_type,
                                    'climb_rate':climb_rate,
                                    'descend_rate':descend_rate,
                                    'start_altitude':start_altitude,
                                    'end_altitude':end_altitude,
                                    'climb_angle':climb_angle,
                                    'descent_angle':descent_angle,
                                    'speed':speed_spec,
                                    'time_required':time_required
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_spec_df.to_csv(base_path+"profile_spec_"+str(profile_id)+'.csv', index=False)

    return mission

def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission


    # done!
    return missions


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results,line_style='bo-'):

    # Plot Flight Conditions
    plot_flight_conditions(results, line_style)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)

    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style)

    # Plot Propeller Conditions
    plot_propeller_conditions(results, line_style)

    # Plot Electric Motor and Propeller Efficiencies
    plot_eMotor_Prop_efficiencies(results, line_style)

    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results, line_style)

    return

if __name__ == '__main__':
    start_time = time.time()
    all_UTM_data_df = pd.read_csv('./logs/all_UTM_sim_data.csv')
    N = all_UTM_data_df.shape[0]
    num_thread = 48
    num_sample_in_thread = N//num_thread
    all_threads = [None]*num_thread
    for i in range(num_thread):
        start_indx = i*num_sample_in_thread
        end_indx = (i+1)*num_sample_in_thread if i+1 != num_thread else N
        all_threads[i] = threading.Thread(target=main, args=(start_indx, end_indx,))
        all_threads[i].start()
        all_threads[i].join()
    
    end_time = time.time()
    print("Total Analysis Time: {}s".format(end_time-start_time))
    
  






    



 


def mission_setup(analyses,vehicle):


    

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                               = Segments.Hover.Climb(base_segment)
    segment.tag                                           = "Climb"
    segment.analyses.extend( analyses.climb)
    segment.altitude_start                                = 0.0  * Units.ft
    segment.altitude_end                                  = 40.  * Units.ft
    segment.climb_rate                                    = 300. * Units['ft/min']
    segment.battery_energy                                = vehicle.networks.battery_propeller.battery.max_energy
    segment.state.unknowns.throttle                       = 0.9 * ones_row(1)
    segment.process.iterate.conditions.stability          = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability       = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment,\
                                                                                         initial_power_coefficient = 0.01)

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Hover Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                                 = Segments.Hover.Hover(base_segment)
    segment.tag                                             = "Hover"
    segment.analyses.extend( analyses.hover )
    segment.altitude                                        = 40.  * Units.ft
    segment.time                                            = 2*60
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)


    # add to misison
    mission.append_segment(segment)

    return mission










if __name__ == '__main__':
    main()
    plt.show(block=True) 
  
