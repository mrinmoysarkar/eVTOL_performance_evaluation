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
sys.path.append('./SUAVE/trunk')
import SUAVE
from SUAVE.Core import Units , Data
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Plots.Geometry_Plots import *
import numpy as np
import pandas as pd
import time

sys.path.append('./SUAVE/regression/scripts/Vehicles')
sys.path.append('./uam_simulator')
# the analysis functions

from performance_analysis import performanceanalyzer

from Stopped_Rotor import vehicle_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    np.random.seed(77)  # Set a random seed to ensure repeatability of a given run
    
    start_time = time.time()
    pa = performanceanalyzer('./logs/example_run.json')
    pa.load_data()
    all_tj = pa.get_all_trajectories()
    count = 0
    for i,tj in enumerate(all_tj):
        # if count == 1:
        #     break
        # count += 1
       # build the vehicle, configs, and analyses
        configs, analyses = full_setup(tj)
        analyses.finalize()
        '''
        # Print weight properties of vehicle
        weights = configs.weight_breakdown
        print(weights)
        print(configs.mass_properties.center_of_gravity)
    
        # check weights
        empty_r       = 831.0480821239719
        structural_r  = 321.68932478738003
        total_r       = 1031.0480821239719
        lift_rotors_r = 16.445392185186808
        propellers_r  = 3.2944573008378044
        prop_motors_r = 2.0
        rot_motors_r  = 36.0
    
        weights_error = Data()
        weights_error.empty       = abs(empty_r - weights.empty)/empty_r
        weights_error.structural  = abs(structural_r - weights.structural)/structural_r
        weights_error.total       = abs(total_r - weights.total)/total_r
        weights_error.lift_rotors = abs(lift_rotors_r - weights.lift_rotors)/lift_rotors_r
        weights_error.propellers  = abs(propellers_r - weights.propellers)/propellers_r
        weights_error.propellers  = abs(prop_motors_r - weights.propeller_motors)/prop_motors_r
        weights_error.propellers  = abs(rot_motors_r - weights.lift_rotor_motors)/rot_motors_r
    
        for k, v in weights_error.items():
            assert (np.abs(v) < 1E-6)
        '''
        # evaluate mission
        mission   = analyses.mission
        results   = mission.evaluate()
    
        # plot results
        # plot_mission(results,configs)
        save_results_as_csv(results,i+1)
    end_time = time.time()
    print("Total Analysis Time: {}s".format(end_time-start_time))

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup(tj):

    # vehicle data
    vehicle  = vehicle_setup()
    plot_vehicle(vehicle,plot_control_points = False)

    # vehicle analyses
    analyses = base_analysis(vehicle)

    # mission analyses
    mission  = mission_setup(analyses,vehicle,tj)

    analyses.mission = mission

    return  vehicle, analyses


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
    noise = SUAVE.Analyses.Noise.Fidelity_One()
    noise.geometry = vehicle
    analyses.append(noise)

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


def mission_setup(analyses,vehicle,tj):

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
    base_segment.state.numerics.number_control_points        = 3
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip

    # VSTALL Calculation
    m      = vehicle.mass_properties.max_takeoff
    g      = 9.81
    S      = vehicle.reference_area
    atmo   = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho    = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax  = 1.2
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))
    print(Vstall)

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

    ######
    # pa = performanceanalyzer('/media/ariac/DATAPART1/mrinmoys-document/UAM_simulator_scitech2021/logs/example_run.json')
    # pa.load_data()
    # tj = pa.plot_trajectories()
    # print(tj.shape)
    total_t = 0
    speed = 0
    for i in range(tj.shape[0]-1):
        total_t = tj[i+1,4] - tj[i,4]
        speed = np.linalg.norm(tj[i,2:4])
    # speed /= tj.shape[0]
    # t = total_t
    # print(Vstall, speed)
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

    return mission



# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results,vec_configs,line_style='bo-'):

    # Plot Flight Conditions
    plot_flight_conditions(results, line_style)

    # Plot Aerodynamic Coefficients
    # plot_aerodynamic_coefficients(results, line_style)

    # Plot Aircraft Flight Speed
    # plot_aircraft_velocities(results, line_style)

    # Plot Aircraft Electronics
    plot_electronic_conditions(results, line_style)

    # Plot Electric Motor and Propeller Efficiencies  of Lift Cruise Network
    # plot_lift_cruise_network(results, line_style)

    return

def load_stopped_rotor_results():
    return SUAVE.Input_Output.SUAVE.load('results_stopped_rotor.res')

def save_stopped_rotor_results(results):

    for segment in results.segments.values():
        del segment.conditions.noise

    SUAVE.Input_Output.SUAVE.archive(results,'results_stopped_rotor.res')
    return

def save_results_as_csv(results, profile_id=0):
    labels = results.segments.keys()
    time_profile = []
    range_profile = []
    energy_profile = []
    c_rating_profile = []
    altitude_profile = []
    voltage_profile = []
    propeller_throttle = []
    lift_throttle = []
    airspeed_profile = []
    label_profile = []
    converged_profile = []
    converged = True
    
    for i in range(len(results.segments)):  
        label = labels[i]
        converged = converged and results.segments[i].converged
        # print(label,results.segments[i].converged)
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min #minuites
        power          = results.segments[i].conditions.propulsion.battery_draw[:,0]  #(Watts)
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]/Units.Wh #W-hr
        volts          = results.segments[i].conditions.propulsion.battery_voltage_under_load[:,0] #volts
        volts_oc       = results.segments[i].conditions.propulsion.battery_voltage_open_circuit[:,0]     
        current        = results.segments[i].conditions.propulsion.battery_current[:,0]      
        battery_amp_hr = (energy/ Units.Wh )/volts  
        C_rating       = current/battery_amp_hr #C-Rate (C)
        eta            = results.segments[i].conditions.propulsion.throttle[:,0]  # propeller motor throttle
        eta_l          = results.segments[i].conditions.propulsion.throttle_lift[:,0] #lift motor throttle
        
        airspeed = results.segments[i].conditions.freestream.velocity[:,0] /   Units['mph']  # speed in mph
        theta    = results.segments[i].conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        
        x        = results.segments[i].conditions.frames.inertial.position_vector[:,0]/ Units.mile #range mile
        y        = results.segments[i].conditions.frames.inertial.position_vector[:,1]
        z        = results.segments[i].conditions.frames.inertial.position_vector[:,2]
        altitude = results.segments[i].conditions.freestream.altitude[:,0]/Units.feet  # feet
        
        
        time_profile += list(time)
        range_profile += list(x)
        energy_profile += list(energy)
        c_rating_profile += list(C_rating)
        altitude_profile += list(altitude)
        voltage_profile += list(volts)
        propeller_throttle += list(eta)
        lift_throttle += list(eta_l) 
        airspeed_profile += list(airspeed)
        label_profile += [label]*len(list(time))
        converged_profile += [results.segments[i].converged] * len(list(time))
    profile_df = pd.DataFrame(data={"time":time_profile,
                                    "range":range_profile,
                                    "energy":energy_profile,
                                    "c_rating":c_rating_profile,
                                    "altitude":altitude_profile,
                                    "voltage":voltage_profile,
                                    "propeller_throttle":propeller_throttle,
                                    "lift_throttle":lift_throttle,
                                    "speed":airspeed_profile,
                                    "label":label_profile,
                                    "converged":converged_profile})
    
    base_path = "/media/ariac/DATAPART1/mrinmoys-document/UAM_simulator_scitech2021/logs/profiles_eval/"
    profile_df.to_csv(base_path+"profile_"+str(profile_id)+"_converged_"+str(converged)+'.csv', index=False)
        
    

if __name__ == '__main__':
    main()
    plt.show(block=True)
