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

from Stopped_Rotor import vehicle_setup, configs_setup

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
            
            # # build the vehicle, configs, and analyses
            # configs, analyses = full_setup(agent_id,tj)
            # analyses.finalize()
            # # evaluate mission
            # mission   = analyses.mission
            # results   = mission.evaluate()
            
            #######
            # build the vehicle, configs, and analyses
            configs, analyses = full_setup(agent_id, tj)
            configs.finalize()
            analyses.finalize()
            # evaluate mission
            mission   = analyses.missions.base
            results   = mission.evaluate()

            # plot results
            plot_mission(results,configs.base)
            ######
            save_results(results, profile_id=agent_id)
            # save_results_as_csv(results, agent_id)
            print(time.time()-start_time_tj, start_indx, end_indx)
    end_time = time.time()
    print("Total Analysis Time: {}s".format(end_time-start_time))
    # num_completed_thread += 1




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
        if tj[i-1,4] > tj[i,4]:
            break
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

def get_random_num(low, high):
    return low + (high-low)*np.random.rand()

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
    mission           = mission_setup(configs_analyses,vehicle, tj, profile_id)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

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


def base_analysis(vehicle):

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
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)


    # ------------------------------------------------------------------
    #  Noise Analysis
    # noise = SUAVE.Analyses.Noise.Fidelity_One()
    # noise.geometry = vehicle
    # analyses.append(noise)

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
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip

    # VSTALL Calculation
    m      = vehicle.mass_properties.max_takeoff
    g      = 9.81
    S      = vehicle.reference_area
    atmo   = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho    = atmo.compute_values(1500.*Units.feet,0.).density
    CLmax  = 1.2
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))


    # ------------------------------------------------------------------
    #   takeoff
    # ------------------------------------------------------------------
    
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "hover_climb"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 50.  * Units.ft
    segment.climb_rate                                       = get_random_num(.7, 1.)*500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.networks.lift_cruise.battery.max_energy
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip  
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_lift_rotor_power_coefficient = 0.01,
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
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag                                         = "dep_transition"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                              = 50.0 * Units.ft
    segment.altitude_end                                = 60.0 * Units.ft
    segment.air_speed                                   = get_random_num(.65, .85) * Vstall
    segment.climb_angle                                 = 6 * Units.degrees
    segment.acceleration                                = 0.5 * Units['m/s/s']
    segment.pitch_initial                               = 5. * Units.degrees
    segment.pitch_final                                 = 7. * Units.degrees
    ones_row                                            = segment.state.ones_row
    segment.state.unknowns.throttle                     = 0.95  * ones_row(1)
    segment.process.iterate.unknowns.mission            = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability     = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_transition_unknowns_and_residuals_to_segment(segment,
                                                         initial_prop_power_coefficient = 0.2,
                                                         initial_lift_rotor_power_coefficient = 0.01,
                                                         initial_throttle_lift = 0.9,)

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
    #   second climb
    # ------------------------------------------------------------------
    segment                                            = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "second_climb"
    segment.analyses.extend( analyses.base )
    segment.air_speed                                  = get_random_num(.9, 1.1)*Vstall
    segment.altitude_start                             = 60.0 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.climb_rate                                 = 500. * Units['ft/min'] 
    segment.state.unknowns.throttle                    =  0.80 * ones_row(1)
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)
    
    # add to misison
    mission.append_segment(segment)
    
    segment_type.append(segment.tag)
    climb_rate.append(segment.climb_rate )
    descend_rate.append(np.nan)
    start_altitude.append(segment.altitude_start)
    end_altitude.append(segment.altitude_end)
    climb_angle.append(np.nan)
    descent_angle.append(np.nan)
    speed_spec.append(segment.air_speed)
    time_required.append(np.nan)
    
    
    
    # ------------------------------------------------------------------
    #   Departure Terminal Procedure
    # ------------------------------------------------------------------
    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag                                        = "departure_terminal_procedures"
    segment.analyses.extend( analyses.base )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = get_random_num(0.7, 1.)*60.   * Units.second
    segment.air_speed                                  = get_random_num(0.95, 1.2)*Vstall
    segment.state.unknowns.throttle                    =  0.80 * ones_row(1)
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
    segment                                          = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag                                      = "accel_climb" 
    segment.analyses.extend( analyses.base) 
    segment.altitude_start                           = 300.0 * Units.ft  
    segment.altitude_end                             = 1500. * Units.ft
    segment.climb_rate                               = 500.  * Units['ft/min']
    segment.air_speed_start                          = get_random_num(0.95, 1.2)*Vstall
    segment.air_speed_end                            = 110.  * Units['mph']    
    segment.state.unknowns.throttle                  = 0.90    *  ones_row(1)
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
    speed_spec.append(np.mean([segment.air_speed_start, segment.air_speed_end]))
    time_required.append(np.nan)
    

    
    total_t = 0
    speed = 0
    for i in range(tj.shape[0]-1):
        total_t = tj[i+1,4] - tj[i,4]
        speed = np.linalg.norm(tj[i,2:4])
        # ------------------------------------------------------------------
        #   Cruise
        # ------------------------------------------------------------------  
        segment                                          = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
        segment.tag                                      = "cruise_"+str(i)
        segment.analyses.extend( analyses.base ) 
        segment.altitude                                 = 1500.0 * Units.ft
        segment.air_speed                                = speed * Units['m/s'] #110.   * Units['mph']
        segment.distance                                 = total_t * speed * Units['m'] #50.    * Units.miles     
        segment.state.unknowns.throttle                  = 0.60 * ones_row(1) 
        segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)     
        
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
        time_required.append(total_t)
        # print("speed {}, time {}".format(speed, total_t))
    
    
    # ------------------------------------------------------------------
    #   Decelerated Descend
    # ------------------------------------------------------------------
    segment                                            = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "decel_descend"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                             = 1500 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.descent_rate                               = 500. * Units['ft/min']
    segment.air_speed                                  = get_random_num(0.95, 1.2) * Vstall
    segment.state.unknowns.throttle                  = 0.90    *  ones_row(1)
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
    segment.analyses.extend( analyses.base )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = get_random_num(0.7, 1.)*60.   * Units.second
    segment.air_speed                                  = get_random_num(0.95, 1.2)*Vstall
    segment.state.unknowns.throttle                    =  0.80 * ones_row(1)
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
    #   second descend
    # ------------------------------------------------------------------
    segment                                             = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                         = "second_descent"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                          = 300.0 * Units.ft
    segment.altitude_end                            = 60.0 * Units.ft
    segment.air_speed                               = get_random_num(.9, 1.1) * Vstall
    segment.descent_rate                            = 500. * Units['ft/min'] 
    segment.state.unknowns.throttle                 =  0.80 * ones_row(1)
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
    #   Transition
    # ------------------------------------------------------------------
    segment                                             = Segments.Descent.Constant_Speed_Constant_Angle(base_segment)
    segment.tag                                         = "app_transition"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                          = 60.0 * Units.ft
    segment.altitude_end                            = 50.0 * Units.ft
    segment.air_speed                               = get_random_num(.65, .85) * Vstall
    segment.descent_angle                           = 6 * Units.degrees
    segment.state.unknowns.throttle                 = 0.95  * ones_row(1)
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
    segment.analyses.extend( analyses.base )
    segment.altitude_start                             = 50.    * Units.ft    
    segment.altitude_end                               = 0.0  * Units.ft  
    segment.descent_rate                               = get_random_num(.7, 1.)*300.  * Units['ft/min']
    segment.process.iterate.unknowns.mission           = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability       = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability    = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_lift_rotor_power_coefficient = 0.01,
                                                                                    initial_throttle_lift = 0.9)
    

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
def plot_mission(results,vec_configs,line_style='bo-'):

    # Plot Flight Conditions
    plot_flight_conditions(results, line_style)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)

    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style)

    # Plot Electric Motor and Propeller Efficiencies  of Lift Cruise Network
    plot_lift_cruise_network(results, line_style)

    return

def save_results(results, profile_id=0):
    #flight conditions
    eVTOL_type = 'lift_and_cruse'
    altitudes = []
    airspeeds = []
    thetas = []
    ranges = []
    times = []
    eVTOL_type_profile = []
    mission_segment_profile = []
    fesibility_profile = []
    for label, segment in zip(results.segments.keys(), results.segments.values()): 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        airspeed = segment.conditions.freestream.velocity[:,0] /   Units['mph']  
        theta    = segment.conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        
        x        = segment.conditions.frames.inertial.position_vector[:,0]/ Units.nmi
        y        = segment.conditions.frames.inertial.position_vector[:,1]
        z        = segment.conditions.frames.inertial.position_vector[:,2]
        altitude = segment.conditions.freestream.altitude[:,0]/Units.feet
        
        altitudes += list(altitude)
        airspeeds += list(airspeed)
        thetas    += list(np.squeeze(theta)) 
        ranges    += list(x)
        times     += list(time)
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        mission_segment_profile += [label]*len(list(time))
        fesibility_profile += [segment.converged] * len(list(time))
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'altitude_ft':altitudes,
                                    'air_speed_mph':airspeeds,
                                    'pitch_angle_deg':thetas,
                                    'range_nmi':ranges,
                                    'time_min':times,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_flight_conditions_"+str(profile_id)+'.csv', index=False)
        
    #Aerodynamic Coefficients
    CLs = []
    CDs = []
    AOAs = []
    L_Ds = []
    times = []
    eVTOL_type_profile = []
    mission_segment_profile = []
    fesibility_profile = []
    for label, segment in zip(results.segments.keys(), results.segments.values()): 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        cl   = segment.conditions.aerodynamics.lift_coefficient[:,0,None] 
        cd   = segment.conditions.aerodynamics.drag_coefficient[:,0,None] 
        aoa  = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd
        
        CLs     += list(np.squeeze(cl))
        CDs     += list(np.squeeze(cd))
        AOAs    += list(aoa) 
        L_Ds    += list(np.squeeze(l_d))
        times   += list(time)
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        mission_segment_profile += [label]*len(list(time))
        fesibility_profile += [segment.converged] * len(list(time))
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'CL':CLs,
                                    'CD':CDs,
                                    'AOA_deg':AOAs,
                                    'L_D':L_Ds,
                                    'time_min':times,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_aerodynamic_coefficients_"+str(profile_id)+'.csv', index=False)
    
    #Aircraft Flight Speed
    velocitys = []
    EASs = []
    machs = []
    times = []
    eVTOL_type_profile = []
    mission_segment_profile = []
    fesibility_profile = []
    for label, segment in zip(results.segments.keys(), results.segments.values()): 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity = segment.conditions.freestream.velocity[:,0] 
        density  = segment.conditions.freestream.density[:,0]
        EAS      = velocity * np.sqrt(density/1.225)
        mach     = segment.conditions.freestream.mach_number[:,0]
        velocity = velocity / Units.kts
        EAS      = EAS / Units.kts
        
        
        velocitys     += list(np.squeeze(velocity))
        EASs          += list(np.squeeze(EAS))
        machs         += list(mach) 
        times         += list(time)
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        mission_segment_profile += [label]*len(list(time))
        fesibility_profile += [segment.converged] * len(list(time))
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'velocity_kts':velocitys,
                                    'equivalent_airspeed_kts':EASs,
                                    'Mach':machs,
                                    'time_min':times,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_aircraft_flight_speed_"+str(profile_id)+'.csv', index=False)
    
    # Aircraft Electronics
    socs    = []
    energys = []
    powers  = []
    voltages = []
    voltages_oc = []
    currents = []
    cratings_instant = []
    cratings_nominal = []
    specific_powers = []
    times = []
    eVTOL_type_profile = []
    mission_segment_profile = []
    fesibility_profile = []
    for label, segment in zip(results.segments.keys(), results.segments.values()): 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        
        pack_power          = segment.conditions.propulsion.battery_power_draw[:,0] 
        pack_energy         = segment.conditions.propulsion.battery_energy[:,0] 
        pack_volts          = segment.conditions.propulsion.battery_voltage_under_load[:,0] 
        pack_volts_oc       = segment.conditions.propulsion.battery_voltage_open_circuit[:,0]     
        pack_current        = segment.conditions.propulsion.battery_current[:,0]   
        pack_SOC            = segment.conditions.propulsion.battery_state_of_charge[:,0]  
        specific_power      = segment.conditions.propulsion.battery_specfic_power[:,0]  
        
        
        pack_battery_amp_hr = (pack_energy/ Units.Wh )/pack_volts  
        pack_C_instant      = pack_current/pack_battery_amp_hr
        pack_C_nominal      = pack_current/np.max(pack_battery_amp_hr)
        
    
        socs             += list(pack_SOC)  
        energys          += list((pack_energy/Units.Wh)/1000)     
        powers           += list(-pack_power/1000)        
        voltages         += list(pack_volts) 
        voltages_oc      += list(pack_volts_oc)
        currents         += list(pack_current)
        cratings_instant += list(pack_C_instant)
        cratings_nominal += list(pack_C_nominal)
        specific_powers  += list(specific_power)
        
        times         += list(time)
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        mission_segment_profile += [label]*len(list(time))
        fesibility_profile += [segment.converged] * len(list(time))
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'SOC':socs,
                                    'battery_energy_kw_h':energys,
                                    'battery_power_kw':powers,
                                    'voltage_v':voltages,
                                    'voltage_oc_v':voltages_oc,
                                    'current_a':currents,
                                    'C_Rating_instant':cratings_instant,
                                    'C_Rating_nominal':cratings_nominal,
                                    'specific_power':specific_powers,
                                    'time_min':times,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_aircraft_electronics_"+str(profile_id)+'.csv', index=False)
    
    # Electric Motor and Propeller Efficiencies
    propeller_throttle = []
    lift_throttle = []
    propeller_rpm = []
    propeller_thrust = []
    propeller_torque = []
    propeller_efficiency = []
    propeller_motor_efficiency = []
    propeller_power_coefficient = []
    lift_rotor_rpms = []
    lift_thrust = []
    lift_torque = []
    lift_efficiency = []
    lift_motor_efficiency = []
    lift_power_coefficient = []
    propeller_tip_mach = []
    lift_tip_mach = []
    
    
    times = []
    eVTOL_type_profile = []
    mission_segment_profile = []
    fesibility_profile = []
    for label, segment in zip(results.segments.keys(), results.segments.values()): 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        
        eta            = segment.conditions.propulsion.throttle[:,0]
        eta_l          = segment.conditions.propulsion.throttle_lift[:,0]
        
        prop_rpm     = segment.conditions.propulsion.propeller_rpm[:,0] 
        prop_thrust  = segment.conditions.frames.body.thrust_force_vector[:,0]
        prop_torque  = segment.conditions.propulsion.propeller_motor_torque[:,0]
        prop_effp    = segment.conditions.propulsion.propeller_efficiency[:,0]
        prop_effm    = segment.conditions.propulsion.propeller_motor_efficiency[:,0]
        prop_Cp      = segment.conditions.propulsion.propeller_power_coefficient[:,0]
        lift_rotor_rpm    = segment.conditions.propulsion.lift_rotor_rpm[:,0] 
        lift_rotor_thrust = -segment.conditions.frames.body.thrust_force_vector[:,2]
        lift_rotor_torque = segment.conditions.propulsion.lift_rotor_motor_torque[:,0]
        lift_rotor_effp   = segment.conditions.propulsion.lift_rotor_efficiency[:,0]
        lift_rotor_effm   = segment.conditions.propulsion.lift_rotor_motor_efficiency[:,0] 
        lift_rotor_Cp     = segment.conditions.propulsion.lift_rotor_power_coefficient[:,0] 
        
        ptm  = segment.conditions.propulsion.propeller_tip_mach[:,0] 
        rtm  = segment.conditions.propulsion.lift_rotor_tip_mach[:,0]
        
        propeller_throttle += list(eta)
        lift_throttle += list(eta_l)
        propeller_rpm += list(prop_rpm)
        propeller_thrust += list(prop_thrust)
        propeller_torque += list(prop_torque)
        propeller_efficiency += list(prop_effp)
        propeller_motor_efficiency += list(prop_effm)
        propeller_power_coefficient += list(prop_Cp)
        lift_rotor_rpms += list(lift_rotor_rpm)
        lift_thrust += list(lift_rotor_thrust)
        lift_torque += list(lift_rotor_torque)
        lift_efficiency += list(lift_rotor_effp)
        lift_motor_efficiency += list(lift_rotor_effm)
        lift_power_coefficient += list(lift_rotor_Cp)
        propeller_tip_mach += list(ptm)
        lift_tip_mach += list(rtm)
        
        
        times         += list(time)
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        mission_segment_profile += [label]*len(list(time))
        fesibility_profile += [segment.converged] * len(list(time))
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'propeller_throttle':propeller_throttle,
                                    'lift_throttle':lift_throttle,
                                    'propeller_rpm':propeller_rpm,
                                    'propeller_thrust_N':propeller_thrust,
                                    'propeller_torque_N_m':propeller_torque,
                                    'propeller_efficiency':propeller_efficiency,
                                    'propeller_motor_efficiency':propeller_motor_efficiency,
                                    'propeller_power_coefficient':propeller_power_coefficient,
                                    'lift_rotor_rpm':lift_rotor_rpms,
                                    'lift_thrust_N':lift_thrust,
                                    'lift_torque_N_m':lift_torque,
                                    'lift_efficiency':lift_efficiency,
                                    'lift_motor_efficiency':lift_motor_efficiency,
                                    'lift_power_coefficient':lift_power_coefficient,
                                    'propeller_tip_mach':propeller_tip_mach,
                                    'lift_tip_mach':lift_tip_mach,
                                    'time_min':times,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_electric_motor_and_propeller_efficiencies_"+str(profile_id)+'.csv', index=False)
    
    
    

def save_results_as_csv(results, profile_id=0):
    labels = results.segments.keys()
    eVTOL_type = 'lift_and_cruse'
    
    eVTOL_type_profile = []
    mission_segment_profile = []
    time_profile = []
    power_profile = []
    energy_profile = []
    volts_profile = []
    volts_oc_profile = []
    current_profile = []
    C_rating_profile = []
    propeller_throttle_profile = []
    lift_throttle_profile = []
    propeller_efficiency_profile = []
    motor_efficiency_profile = []
    lift_profile = []
    drag_profile = []
    mass_profile = []
    thrust_profile = []
    speed_profile = []
    air_speed_profile = []
    propeller_disk_loading_profile = []
    peopeller_power_loading_profile = []
    lift_disk_loading_profile = []
    lift_power_loadind_profile = []
    mach_number_profile = []
    AoA_profile = []
    CL_profile = []
    CD_profile = []
    L_D_profile = []
    x_profile = []
    y_profile = []
    altitude_profile = []
    roll_profile = []
    pitch_profile = []
    yaw_profile = []
    fesibility_profile = []
    
    converged = True
    
    for i in range(len(results.segments)):  
        label = labels[i]
        converged = converged and results.segments[i].converged
        # print(label,results.segments[i].converged)
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min #minuites
        time_profile += list(time)
        
        eVTOL_type_profile += [eVTOL_type]*len(list(time))
        
        mission_segment_profile += [label]*len(list(time))
        
        power          = -results.segments[i].conditions.propulsion.battery_draw[:,0]  #(Watts)
        power_profile += list(power)
        
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]/Units.Wh #W-hr
        energy_profile += list(energy)
        
        volts          = results.segments[i].conditions.propulsion.battery_voltage_under_load[:,0] #volts
        volts_profile += list(volts)
        
        volts_oc       = results.segments[i].conditions.propulsion.battery_voltage_open_circuit[:,0]
        volts_oc_profile += list(volts_oc)
        
        current        = results.segments[i].conditions.propulsion.battery_current[:,0] 
        current_profile += list(current)
        
        battery_amp_hr = (energy/ Units.Wh )/volts  
        C_rating       = current/battery_amp_hr #C-Rate (C)
        C_rating_profile += list(C_rating)
        
        eta            = results.segments[i].conditions.propulsion.throttle[:,0]  # propeller motor throttle
        propeller_throttle_profile += list(eta)
        
        eta_l          = results.segments[i].conditions.propulsion.throttle_lift[:,0] #lift motor throttle
        lift_throttle_profile += list(eta_l)
        
        effp   = results.segments[i].conditions.propulsion.etap[:,0] # 'Propeller Efficiency ($\eta_p$)'
        propeller_efficiency_profile += list(effp)
        
        effm   = results.segments[i].conditions.propulsion.etam[:,0] # 'Motor Efficiency ($\eta_m$)'
        motor_efficiency_profile += list(effm)
        
        # prop_rpm     = results.segments[i].conditions.propulsion.propeller_rpm[:,0] 
        # prop_thrust  = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        # prop_torque  = results.segments[i].conditions.propulsion.propeller_motor_torque[:,0]
        # prop_effp    = results.segments[i].conditions.propulsion.propeller_efficiency[:,0]
        # prop_effm    = results.segments[i].conditions.propulsion.propeller_motor_efficiency[:,0]
        # prop_Cp      = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
        # lift_rotor_rpm    = results.segments[i].conditions.propulsion.lift_rotor_rpm[:,0] 
        # lift_rotor_thrust = -results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        # lift_rotor_torque = results.segments[i].conditions.propulsion.lift_rotor_motor_torque[:,0]
        # lift_rotor_effp   = results.segments[i].conditions.propulsion.lift_rotor_efficiency[:,0]
        # lift_rotor_effm   = results.segments[i].conditions.propulsion.lift_rotor_motor_efficiency[:,0] 
        # lift_rotor_Cp     = results.segments[i].conditions.propulsion.lift_rotor_power_coefficient[:,0] 
        
       
        Lift   = -results.segments[i].conditions.frames.wind.lift_force_vector[:,2] # in newton
        lift_profile += list(Lift)
        
        Drag   = -results.segments[i].conditions.frames.wind.drag_force_vector[:,0]   # in newton    
        drag_profile += list(Drag)
        
        mass     = results.segments[i].conditions.weights.total_mass[:,0] / Units.lb
        mass_profile += list(mass)
        
        thrust   =  results.segments[i].conditions.frames.body.thrust_force_vector[:,0] # in newton
        thrust_profile += list(thrust)
        
        speed = results.segments[i].conditions.freestream.velocity[:,0] /   Units['mph']  # speed in mph
        speed_profile += list(speed)
        
        density  = results.segments[i].conditions.freestream.density[:,0]
        equivalent_air_speed      = speed * np.sqrt(density/1.225)
        air_speed_profile += list(equivalent_air_speed)
        
        propeller_Disk_Loading    = results.segments[i].conditions.propulsion.propeller_disc_loading[:,0] #'lift disc power N/m^2'
        propeller_disk_loading_profile += list(propeller_Disk_Loading)
        
        propeller_Power_Loading    = results.segments[i].conditions.propulsion.propeller_power_loading[:,0] #'lift power loading (N/W)'
        peopeller_power_loading_profile += list(propeller_Power_Loading)
        
        lift_Disk_Loading    = results.segments[i].conditions.propulsion.lift_rotor_disc_loading[:,0]
        lift_disk_loading_profile += list(lift_Disk_Loading)
        
        lift_Power_Loading    = results.segments[i].conditions.propulsion.lift_rotor_power_loading[:,0] 
        lift_power_loadind_profile += list(lift_Power_Loading)
        
        
        
        
        mach_number = results.segments[i].conditions.freestream.mach_number[:,0]
        mach_number_profile += list(mach_number)
        
        aoa = results.segments[i].state.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        AoA_profile += list(aoa)
        
        lift_coeff = results.segments[i].state.conditions.aerodynamics.lift_coefficient[:,0]
        CL_profile += list(lift_coeff)
        
        drag_coeff = results.segments[i].state.conditions.aerodynamics.drag_coefficient[:,0]
        CD_profile += list(drag_coeff)
        
        lift_by_drag = lift_coeff/drag_coeff
        L_D_profile += list(lift_by_drag)
        
        x        = results.segments[i].conditions.frames.inertial.position_vector[:,0]/ Units.mile #range mile
        x_profile += list(x)
        
        y        = results.segments[i].conditions.frames.inertial.position_vector[:,1]/ Units.mile
        y_profile += list(y)
        
        
        # z        = results.segments[i].conditions.frames.inertial.position_vector[:,2]
        altitude = results.segments[i].conditions.freestream.altitude[:,0]/Units.feet  # feet
        altitude_profile += list(altitude)
        
        roll    = results.segments[i].conditions.frames.body.inertial_rotations[:,0,None] / Units.deg
        roll_profile += list(roll)
        
        pitch    = results.segments[i].conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        pitch_profile += list(pitch) 
        
        yaw    = results.segments[i].conditions.frames.body.inertial_rotations[:,2,None] / Units.deg
        yaw_profile += list(yaw)
        
        fesibility_profile += [results.segments[i].converged] * len(list(time))
        
        
    profile_df = pd.DataFrame(data={'eVTOL_type':eVTOL_type_profile,
                                    'mission_segment':mission_segment_profile,
                                    'time':time_profile,
                                    'power':power_profile,
                                    'energy':energy_profile,
                                    'volts':volts_profile,
                                    'volts_oc':volts_oc_profile,
                                    'current':current_profile,
                                    'C_rating':C_rating_profile,
                                    'propeller_throttle':propeller_throttle_profile,
                                    'lift_throttle':lift_throttle_profile,
                                    'propeller_efficiency':propeller_efficiency_profile,
                                    'motor_efficiency':motor_efficiency_profile,
                                    'lift':lift_profile,
                                    'drag':drag_profile,
                                    'mass':mass_profile,
                                    'thrust':thrust_profile,
                                    'speed':speed_profile,
                                    'air_speed':air_speed_profile,
                                    'propeller_disk_loading':propeller_disk_loading_profile,
                                    'peopeller_power_loading':peopeller_power_loading_profile,
                                    'lift_disk_loading':lift_disk_loading_profile,
                                    'lift_power_loadind':lift_power_loadind_profile,
                                    'mach_number':mach_number_profile,
                                    'AoA':AoA_profile,
                                    'CL':CL_profile,
                                    'CD':CD_profile,
                                    'L_D':L_D_profile,
                                    'x':x_profile,
                                    'y':y_profile,
                                    'altitude':altitude_profile,
                                    'roll':roll_profile,
                                    'pitch':pitch_profile,
                                    'yaw':yaw_profile,
                                    'fesibility':fesibility_profile
                                    })
    
    base_path = "./logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_"+str(profile_id)+'.csv', index=False)
        
    

if __name__ == '__main__':
    main(0,1)
    plt.show(block=True)
    # start_time = time.time()
    # all_UTM_data_df = pd.read_csv('./logs/sampled_UTM_dataset.csv')
    # all_UTM_data_df = all_UTM_data_df[all_UTM_data_df['eVTOL_type']=='lift_and_cruse']
    # N = all_UTM_data_df.shape[0]
    # num_thread = 48
    # num_sample_in_thread = N//num_thread
    # all_threads = [None]*num_thread
    # num_completed_thread = 0
    # for i in range(num_thread):
    #     start_indx = i*num_sample_in_thread
    #     end_indx = (i+1)*num_sample_in_thread if i+1 != num_thread else N
    #     # main(start_indx, end_indx)
    #     all_threads[i] = threading.Thread(target=main, args=(start_indx, end_indx,))
    #     all_threads[i].start()
    #     # all_threads[i].join()
    # while num_completed_thread != num_thread:
    #     time.sleep(30)
    
    # end_time = time.time()
    # print("Total Analysis Time: {}s".format(end_time-start_time))
    
    
