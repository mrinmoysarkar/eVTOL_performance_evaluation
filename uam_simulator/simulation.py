import uam_simulator.environment as environment
import uam_simulator.centralized_manager as centralized_manager
from uam_simulator.my_utils import MyLogger
from threading import Thread
import numpy as np
import heapq
import json
import random


class Simulation:
    def __init__(self, length, n_intruder, minimum_separation, max_speed,
                 time_step= 10, time_end=6000, n_valid_agents_for_simulation_end=None, sensing_radius=None,
                 simulation_type='reactive', simulation_name='test', algorithm_type='MVP',
                 structure=None,
                 multiple_planning_agents=True,
                 log_type='short',save_file=True,
                 log_density=False, update_queue=None,stdout_to_file=False, location='ATL', month='January'):
        """
            length: simulated area length
            n_intruder: number of agents that should be present at the same time in the simulation area
            minimum_separation: desired minimum separation distance between agents
            max_speed: max_speed in m/s of the agents
            time_step: how long should be the simulation time step in seconds
            time_end: stops the simulation after time_end seconds have elapsed, set to float('inf') if using another criteria to stop simulating
            n_valid_agents_for_simulation_end: Valid agents are agents that were created after the time (t0) when all agents in the simulation have been created after the desired number of agents are in the simulation area (t_density)
                                               Once n_valid_agents have finished their flight the simulation ends (unless time_end comes first)
            simulation_type: reactive or strategic
            simulation_name: for saving
            algorithm_type: collision avoidance/ preflight planning algorithm
            structure: specify layers and their range
            demand: dictionary, if type is population density link to a file containing location of start and goals and how often they should be selected
            multiple_planning_agents: set to True if all agents plan, set to False if only one agent plans (debugging mode for planning algo)
            log_type: 'short' means only logging smart agents (only makes a difference if multiple_planning_agents is set to False)
            save_file: if you want to save in the thread set it to true, useful for one off, set to False for batch
            log_density: looking at density of agents in the simulation
            Update Queue: the queue holding agents position for display purposes (set to None if no need to display)
        """
        self.update_queue = update_queue
        self.dt = time_step
        self.time_end = time_end
        self.time = 0.0
        # Tolerance to remove agents from simulation when within tolerance of its goal
        self.tolerance = 1000               # max_speed * self.dt
        self.minimum_separation = minimum_separation
        if sensing_radius is None:
            self.sensing_radius = 2*minimum_separation*10
        else:
            self.sensing_radius = sensing_radius
        self.max_speed = max_speed
        self.log_type = log_type
        self.structure = structure
        if algorithm_type == 'SIPP':
            self.centralized_manager = centralized_manager.SIPPCentralizedManager(minimum_separation, length)
        elif algorithm_type == 'LocalVO':
            self.centralized_manager = centralized_manager.VOCentralizedManager(minimum_separation)
        else:
            self.centralized_manager = centralized_manager.CentralizedManager(minimum_separation)
        if simulation_type == 'strategic':
            utm_on = True
        else:
            utm_on = False
        if stdout_to_file:
            self.my_logger = MyLogger(simulation_name + '.txt')
        else:
            self.my_logger = MyLogger()
        self.demand = None
        self.env = environment.Environment(length, minimum_separation, max_speed, self.sensing_radius, self.tolerance, utm_on,
                                           desired_number_of_agents=n_intruder, multiple_planning_agents=multiple_planning_agents,
                                           structure=structure,
                                           centralized_manager=self.centralized_manager, simulation_type=simulation_type,
                                           algo_type=algorithm_type, log_type=log_type, log_density=log_density,
                                           n_valid_agents_for_simulation_end=n_valid_agents_for_simulation_end, logger=self.my_logger,
                                           location=location, month=month)

        self.length = length
        self.n_intruder = n_intruder
        self.record = {}
        self.simulation_name = simulation_name
        self.simulation_type = simulation_type
        self.algorithm_type = algorithm_type
        self.multiple_planning_agents = multiple_planning_agents
        self.save_file=save_file
        self.logs = None

    def create_constant_density_random_agents(self):
        """ Creates random agent for the length of the simulation and ensures that at any time the number of simulated
        agent is constant (useful if there is only one smart agent and all flight plans must be created before it
        starts)"""
        end_times = []
        # Create the initial n agents everywhere in the sim area, store their predicted(=actual) end time in a heap
        for i in range(0, self.n_intruder):
            agent = self.env.add_random_agent(random_type='in map')
            time = agent.get_predicted_end_time()
            heapq.heappush(end_times, time)
        new_time = heapq.heappop(end_times)
        # For every flight that finishes at time t, a new flight is created that starts at time t
        while new_time < self.time_end:
            agent = self.env.add_random_agent(random_type='edge',start_time=new_time)
            time = agent.get_predicted_end_time()
            new_time = heapq.heappushpop(end_times,time)

    def _run(self):
        # The first element of the update queue will not be drawn properly due to the canvas initialization
        # This can be solved with an empty element at the beginning
        if self.update_queue is not None:
            self.update_queue.put({})
        self.my_logger.log('simulation is starting')
        # An intruder that will collide with the smart agent if nothing is done
        # intruder_start = np.asarray([0.0, self.length], dtype=np.float32)
        # intruder_end = np.asarray([self.length, 0.0], dtype=np.float32)
        # self.env.add_intruder(intruder_start, intruder_end)
        if not self.multiple_planning_agents:
            if self.simulation_type == 'reactive':
                for i in range(0, self.n_intruder):
                    self.env.add_random_agent(random_type='in map')
                if self.algorithm_type != 'empty':
                    self.env.add_reactive_agent(np.asarray([1000.0, 1000.0], dtype=np.float32),
                                                np.asarray([self.length-1000, self.length-1000], dtype=np.float32),
                                                algo_type=self.algorithm_type)
            elif self.simulation_type == 'strategic':
                self.create_constant_density_random_agents()
                # self.env.add_intruder(np.array([19000, 19000.0]),np.array([1000,1000]),0)
                if self.algorithm_type != 'empty':
                    # self.env.add_strategic_agent(np.asarray([1000.0, 1000.0], dtype=np.float32),
                    #                              np.asarray([self.length-1000, self.length-1000], dtype=np.float32),
                    #                              algo_type=self.algorithm_type)
                    self.env.add_strategic_agent(algo_type=self.algorithm_type)
        else:
            if self.algorithm_type != 'empty':
                if self.simulation_type == 'reactive':
                    self.env.add_reactive_agent(algo_type=self.algorithm_type)
                elif self.simulation_type == 'strategic':
                    self.env.add_strategic_agent(algo_type=self.algorithm_type)
        finished=False
        while self.time < self.time_end and not finished:
            finished = self.env.run(self.time, self.dt)
            if self.algorithm_type == 'empty':
                finished=False
            if self.update_queue is not None:
                agents_display = {}
                for agent in self.env.smart_agents+self.env.dumb_agents:
                    if self.time >= agent.start_time:
                        agents_display[agent.id] = {'x': agent.position[0],
                                                     'y': agent.position[1],
                                                     'heading': agent.heading,
                                                     'radius': agent.radius,
                                                     'status': agent.status,
                                                     'ownship': agent.ownship}
                for agent in self.env.phantom_agents:
                    agents_display[agent.id] = {'x': agent.position[0],
                                                'y': agent.position[1],
                                                'heading': agent.heading,
                                                'radius': agent.radius,
                                                'status': agent.status,
                                                'ownship': agent.ownship}
                self.update_queue.put(agents_display)
            self.time += self.dt
        if finished:
            self.my_logger.log('environment sent termination signal')
        if self.time >= self.time_end:
            self.my_logger.log('simulation time finished')
        log_data = self.env.terminate()
        log_data['inputs'] = self.log_inputs()
        filename = 'logs/' + self.simulation_name + '.json'
        if self.save_file:
            with open(filename,'w') as file:
                json.dump(log_data, file, indent=4)
        self.logs = log_data
        self.my_logger.remove_logfile()

    def run(self):
        self._run()
        # thread = Thread(target=self._run)
        # thread.start()
        # If saving in the _run method, no need to join the thread
        # If not, then join the thread to wait for the logs
        # if not self.save_file:
        #     thread.join()
        # return self.logs

    def log_inputs(self):
        """ In the log file we need to keep the parameters of that particular simulation """
        input_dic = {'area_length_m': self.length, 'n_intruders': self.n_intruder,
                     'h_collision_dist_m': self.minimum_separation, 'max_speed_m_s': self.max_speed,
                     'sensing_radius': self.sensing_radius,
                     'sim_time_step_s': self.dt, 'sim_length_s': self.time_end,
                     'simulation_type': self.simulation_type, 'algorithm_type': self.algorithm_type,
                     'structure': self.structure
                     }
        if self.demand is not None:
            input_dic['demand'] = self.demand
        return input_dic

