import numpy as np
import pandas as pd
import random
from uam_simulator import agent
from uam_simulator.my_utils import MyLogger
import math
import json
from sklearn.neighbors import KDTree

import sys
sys.path.append('./uam_simulator')

from weather_manager import weather_manager

class Environment:
    def __init__(self, size_map, min_distance, agent_speed, sensing_radius, tolerance, utm_on, desired_number_of_agents=1,centralized_manager=None,
                 multiple_planning_agents=False, structure=None, simulation_type='reactive', algo_type='A_star_8', log_type='short', log_density=False,
                 n_valid_agents_for_simulation_end=None,logger=MyLogger(), location='ATL', month='January'):
        # should add a buffer around the environment to prevent initialization glitches
        # Or pop some opposing aircraft in the middle
        # Requires agent speed to create the UTM (not very elegant, need to find a different solution, but will do for now)
        self.my_logger=logger
        self.total_number_of_agents = 0
        # self.agents = []
        self.agent_pos = []
        self.agent_record_list = []
        self.conflicts_list = []
        self.conflicts = set()
        self.conflicts_dic = {}
        self.size_map = size_map
        self.speed = agent_speed
        self.sensing_radius = sensing_radius
        self.smart_agents = []
        self.dumb_agents = []
        self.active_agents = []
        self.active_agent_pos = []
        self.phantom_agents = []
        self.waiting_agent = []
        self.centralized_manager = None
        self.utm_on = utm_on
        self.simulation_type = simulation_type
        self.log_type = log_type
        self.log_density = log_density
        if self.log_density:
            self.density_map = DensityMap(self.size_map, 500)
        if utm_on:
            self.centralized_manager = centralized_manager
            self.default_planning_time_step = 2 * min_distance / self.speed
        self.time = 0.0
        self.min_distance = min_distance
        self.tolerance = tolerance
        self.dt = 0  # for debugging purposes
        self.multiple_planning_agents = multiple_planning_agents
        self.desired_number_of_agents = desired_number_of_agents
        self.structure = structure
        self.demand = None
        self.lz_coordinates = None
        self.cumulative_densities = None
        self.algo_type=algo_type
        self.agent_added_time = 0
        self.ramping_up = True
        self.agents_kd_tree = None
        self.t0 = None # Time where all agents in the simulation have been created after the target density was reached
        self.n_valid_density_agents = 0
        self.t_density = None
        self.n_min_valid_agents = n_valid_agents_for_simulation_end
        self.debug_time =0
        self.agent_oldest_time=0
        self.vertiport_db = None
        self.weather_manager = weather_manager('./weather_data/Summarized_Wind_Statistics.csv')
        self.location = location
        self.month = month

    def run(self, time, sim_dt):
        # Agents decide what their next move is
        # Then everyone moves
        # This way the order at which agents are updated does not matter
        # At the beginning the time is time
        # At the end the time is time+sim_dt
        self.time = time
        self.dt = sim_dt  # for debugging purposes
        if self.time-self.debug_time >= 600:
            self.my_logger.log('sim time is '+str(time)+', density is '+str(len(self.smart_agents)+len(self.dumb_agents)))
            self.my_logger.log('Agent oldest date of creation: '+str(self.agent_oldest_time))
            self.debug_time = self.time
        waiting_agents_ready_to_take_off = []
        for a in self.waiting_agent:
            if a.can_safely_take_off(self.time):
                waiting_agents_ready_to_take_off.append(a)
        for a in waiting_agents_ready_to_take_off:
            # Assume only smart agents can wait on the ground
            self.waiting_agent.remove(a)
            self.smart_agents.append(a)
        density = len(self.smart_agents)+len(self.dumb_agents)
        for a in self.smart_agents+self.dumb_agents:
            if self.time >= a.start_time:
                a.compute_next_move(self.time, sim_dt, density=density)
        for a in self.phantom_agents:
            a.compute_next_move(self.time, sim_dt)
        self.active_agents = []
        self.active_agent_pos = []
        for a in self.smart_agents+self.dumb_agents:
            if self.time >= a.start_time:
                self.active_agents.append(a)
                pos = a.move()
                self.active_agent_pos.append(pos)
        for a in self.phantom_agents:
            a.move()
        if self.log_density:
            self.update_density()
        self.add_agents(self.time+self.dt)
        self.remove_outsiders(self.time+self.dt)
        self.build_kd_tree()
        self.check_collisions(self.time+self.dt)
        if self.ramping_up:
            if len(self.smart_agents)+len(self.dumb_agents) >= self.desired_number_of_agents:
                self.ramping_up = False
                self.t_density=self.time+self.dt
                self.my_logger.log('environment reached the desired density of agents at '+str(self.t_density))
        if not self.multiple_planning_agents and self.smart_agents == [] and self.waiting_agent == []:
            finished = True
        elif self.n_min_valid_agents is not None and self.n_valid_density_agents >= self.n_min_valid_agents:
            finished = True
        else:
            finished = False
        return finished

    def get_random_start_and_end(self, protected_area_start=None):
        if self.demand is None:
            # return self.get_random_start_and_end_on_edge(protected_area_start=protected_area_start)
            return self.get_vertiport_start_and_end()
        else:
            if self.demand['type'] == 'hub_and_spoke':
                return self.get_start_and_end_with_hub_and_spoke(protected_area_start=protected_area_start)
            elif self.demand['type'] == 'population_density':
                return self.get_start_and_end_with_pop_density(protected_area_start=protected_area_start)
            else:
                self.my_logger.log('This type of demand is not implemented '+self.demand['type'])

    def get_vertiport_start_and_end(self):
        if self.vertiport_db is None:
            self.vertiport_db = pd.read_csv("./config/vertiport_db.csv")
        vertiport_ids = list(map(int, self.vertiport_db['vertiport_id']))
        ids = np.random.choice(vertiport_ids, size=2, replace=False)
        
        start = np.array([self.vertiport_db.iloc[ids[0]-1].x, self.vertiport_db.iloc[ids[0]-1].y])
        end = np.array([self.vertiport_db.iloc[ids[1]-1].x, self.vertiport_db.iloc[ids[1]-1].y])
        return start, end, ids[0]-1, ids[1]-1
        
        
    def get_random_start_and_end_on_edge(self, protected_area_start=None):
        valid_structure = False
        valid_start_and_end = False
        n_trials_structure = 0
        max_trials_structure = 100
        while (not valid_structure or not valid_start_and_end) and n_trials_structure < max_trials_structure:
            a = random.randint(1, self.size_map - 1)  # +1, -1 to avoid corner cases
            # b = random.randint(0, self.size_map)
            side_a = random.randint(0, 3)
            # side_b = random.randint(1, 3)
            # depending on the side picked build the start vector
            start_x = a * (-(side_a % 2) + 1) + (side_a == 1) * self.size_map
            start_y = a * (side_a % 2) + (side_a == 2) * self.size_map
            start = np.array([float(start_x), float(start_y)])
            if self.algo_type == 'SIPP':
                i, j = self.centralized_manager.get_coordinates(start)
                start = self.centralized_manager.get_position(i, j)
            ## Check if there is an issue with a protected area
            if protected_area_start is not None:
                # This prevents a dumb agent from being initialized too close to a reactive agent
                number_of_trials = 0
                max_number_of_trials = 10
                while np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius'] and number_of_trials < max_number_of_trials:
                    a = random.randint(0, self.size_map)
                    # b = random.randint(0, self.size_map)
                    start_x = a * (-(side_a % 2) + 1) + (side_a == 1) * self.size_map
                    start_y = a * (side_a % 2) + (side_a == 2) * self.size_map
                    number_of_trials += 1
                    start = np.array([float(start_x), float(start_y)])
                if np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius']:
                    print(start)
                    print(protected_area_start['center'])
                    self.my_logger.log('getRandomStartAndEnd failed to place the random agent in a conflict free zone')
            # This initial process did not result in a uniform distribution of angle
            # side_b = (side_a + side_b) % 4
            # end_x = b * (-(side_b % 2) + 1) + (side_b == 1) * self.size_map
            # end_y = b * (side_b % 2) + (side_b == 2) * self.size_map
            # end = np.array([float(end_x), float(end_y)])
            # This should result in a uniform distribution
            max_number_of_trials = 10
            number_of_trials = 0
            valid_start_and_end = False
            while not valid_start_and_end and number_of_trials < max_number_of_trials:
                angle = random.random() * math.pi
                end_x = None
                end_y = None
                if angle == 0:  # probability 0
                    # Returns a corner of the line where it started
                    end_x = (side_a == 1) * self.size_map
                    end_y = (side_a == 2) * self.size_map
                elif angle == (math.pi / 2):
                    side_b = (side_a + 2) % 4
                    end_x = a * (-(side_b % 2) + 1) + (side_b == 1) * self.size_map
                    end_y = a * (side_b % 2) + (side_b == 2) * self.size_map
                else:
                    # compute the intersection with all three other sides (catch exception if angle is pi/2)
                    # Also exception if we are exactly at the corner
                    for i in range(1, 4):
                        side_b = (side_a + i) % 4
                        if (side_b % 2) == 1:
                            x = (side_b == 1) * self.size_map
                            y = start_y + math.tan(angle) * (x - start_x)
                            if 0 <= y <= self.size_map and x != start_x:
                                my_side = i
                                end_x = x
                                end_y = y
                        else:
                            y = (side_b == 2) * self.size_map
                            x = start_x + (1 / math.tan(angle)) * (y - start_y)
                            if 0 <= x <= self.size_map and y != start_y:
                                my_side = i
                                end_x = x
                                end_y = y
                if end_x is None or end_y is None:
                    print('environment random start and end bug')
                    print(angle)
                    print(start_x)
                    print(start_y)
                    print('side a, ', side_a)
                    print('side b, ', my_side)

                end = np.array([float(end_x), float(end_y)])
                if self.algo_type == 'SIPP':
                    i, j = self.centralized_manager.get_coordinates(end)
                    end = self.centralized_manager.get_position(i, j)
                # Is the pair valid ?
                if np.linalg.norm(end-start) > self.tolerance:
                    valid_start_and_end = True
                number_of_trials += 1
            if number_of_trials >= max_number_of_trials:
                print('get random start and end failed to find a valid pair')
            if self.structure is None:
                valid_structure = True
            else:
                if self.structure['type'] == 'layer':
                    heading = math.atan2(end[1] - start[1], end[0] - start[0]) * 180 / math.pi
                    if self.structure['parameters'][1] > heading >= self.structure['parameters'][0]:
                        valid_structure = True
                n_trials_structure += 1
        if n_trials_structure >= max_trials_structure:
            print('get random start and end Failed to find a pair valid for the structure ')
        return start, end

    def get_start_and_end_with_pop_density(self, protected_area_start=None):
        if self.lz_coordinates is None:
            if self.demand['type'] == 'population_density':
                with open(self.demand['parameters']) as f:
                    data = json.load(f)
                    self.lz_coordinates = np.array(data['coordinates_xy'], dtype=np.float64)
                    self.cumulative_densities = data['cumulative_distribution']
            else:
                print('this demand type is not implemented '+str(self.demand))

        valid_structure = False
        valid_start_and_end = False
        n_trials_structure = 0
        max_trials_structure = 100
        while (not valid_structure or not valid_start_and_end) and n_trials_structure < max_trials_structure:
            # Select start
            val1 = random.random()
            index1 = np.searchsorted(self.cumulative_densities, val1)
            start = self.lz_coordinates[index1]
            if protected_area_start is not None:
                number_of_trials = 0
                max_number_of_trials = 10
                while np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius'] and number_of_trials < max_number_of_trials:
                    number_of_trials += 1
                    val1 = random.random()
                    index1 = np.searchsorted(self.cumulative_densities, val1)
                    start = self.lz_coordinates[index1]
                if np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius']:
                    print(start)
                    print(protected_area_start['center'])
                    self.my_logger.log('getRandomStartAndEnd failed to place the random agent in a conflict free zone')
            # Select goal
            valid_start_and_end = False
            max_number_of_trials=10
            number_of_trials = 0
            while not valid_start_and_end and number_of_trials < max_number_of_trials:
                number_of_trials += 1
                val2 = random.random()
                index2 = np.searchsorted(self.cumulative_densities, val2)
                goal = self.lz_coordinates[index2]
                if np.linalg.norm(start-goal) > self.tolerance:
                    valid_start_and_end = True
            # Check the structure to ensure that start and end
            if self.structure is None:
                valid_structure = True
            else:
                if self.structure['type'] == 'layer':
                    heading = math.atan2(goal[1] - start[1], goal[0] - start[0]) * 180 / math.pi
                    if self.structure['parameters'][1] > heading >= self.structure['parameters'][0]:
                        valid_structure = True
                    n_trials_structure += 1
        if n_trials_structure >= max_trials_structure:
            print('get_start_and_end_with_demand failed to find a pair valid for the structure')
        return start, goal

    def get_start_and_end_with_hub_and_spoke(self, protected_area_start=None):
        if self.lz_coordinates is None:
            if self.demand['type']== 'hub_and_spoke':
                with open(self.demand['parameters']) as f:
                    data=json.load(f)
                    self.lz_coordinates = {}
                    for k in data:
                        self.lz_coordinates[int(k)] = {'distribution_center': np.array(data[k]['distribution_center'], dtype=np.float64),
                                                       'customers': np.array(data[k]['customers'], dtype=np.float64)}

        valid_start_and_end = False
        max_number_of_trials = 100
        n_trials = 0
        while not valid_start_and_end and n_trials < max_number_of_trials:
            n_trials += 1
            index_start = random.randrange(0, len(self.lz_coordinates))
            start = self.lz_coordinates[index_start]['distribution_center']
            if protected_area_start is not None:
                number_of_trials = 0
                max_number_of_trials = 10
                while np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius'] and number_of_trials < max_number_of_trials:
                    number_of_trials += 1
                    index_start = random.randrange(0, len(self.lz_coordinates))
                    start = self.lz_coordinates[index_start]['distribution_center']
                if np.linalg.norm(protected_area_start['center'] - start) <= protected_area_start['radius']:
                    print(start)
                    print(protected_area_start['center'])
                    self.my_logger.log('getRandomStartAndEnd failed to place the random agent in a conflict free zone')
            index_end = random.randrange(0, len(self.lz_coordinates[index_start]['customers']))
            goal = self.lz_coordinates[index_start]['customers'][index_end]
            if np.linalg.norm(start - goal) > self.tolerance:
                if self.structure is None:
                    valid_start_and_end = True
                else:
                    if self.structure['type'] == 'layer':
                        heading = math.atan2(goal[1] - start[1], goal[0] - start[0]) * 180 / math.pi
                        if self.structure['parameters'][1] > heading >= self.structure['parameters'][0]:
                            valid_start_and_end = True
        if n_trials >= max_number_of_trials:
            self.my_logger.log('exceeded max number of trials to place random start and end for hub and spoke')
        return start, goal

    def check_collisions(self, time):
        # Naive implementation
        # The measure area should be inside the experimentation area with some margin to avoid the case where an intruder is created right next to the ownship
        current_conflicts = set()
        for a in self.active_agents:
            a.status = 'ok'
        for i in range(0,len(self.active_agents)):
            agentA = self.active_agents[i]
            # Multiplying the distance by a factor slightly smaller than one to avoid numerical error
            agents = self.get_neighbors(agentA.position, self.min_distance*0.99999)
            for agentB in agents:
                if agentB.id > agentA.id:
                    current_conflicts.add(frozenset([agentA, agentB]))  # Guarantees that each conflict is counted once
                    agentA.status = 'boom'
                    agentB.status = 'boom'
                    # print('Position of conflict, agent A and B : '+str(agentA.position)+', '+str(agentB.position))
                    # print('Flight plans A '+str(agentA.flightPlan.positions))
                    # print(agentA.flightPlan.times)
                    # print('Flight plans B ' + str(agentB.flightPlan.positions))
                    # print(agentB.flightPlan.times)
        # Conflicts that are over
        finished_conflicts = self.conflicts - current_conflicts
        for c in finished_conflicts:
            conflict_object = self.conflicts_dic.pop(c)
            self.conflicts_list.append(conflict_object.bundle_conflict())
        continued_conflicts = self.conflicts & current_conflicts
        for c in continued_conflicts:
            conflict_object = self.conflicts_dic[c]
            conflict_object.update(time)
        started_conflicts = current_conflicts - self.conflicts
        for c in started_conflicts:
            conflict_object = Conflict(time, c)
            self.conflicts_dic[c] = conflict_object
            conflict_object.update(time)
        self.conflicts = current_conflicts

    def remove_outsiders(self, time):
        """Remove agents near their goal, also check if all agents in the simulation have been created after the density was reached"""
        agents_to_remove = []
        found_old_agents=False
        agent_oldest_time = time
        for a in self.smart_agents+self.dumb_agents:
            # TODO: within 1km of the goal test
            dist_to_goal = np.linalg.norm(a.position - a.goal)
            if a.start_time < agent_oldest_time:
                agent_oldest_time = a.start_time
            if dist_to_goal <= self.tolerance:
                a.finish_flight(time+dist_to_goal/a.maxSpeed, goal_pos=a.goal, t_removed_from_sim=time)
                if self.log_type != 'short' or a.agent_logic != 'dumb':
                    self.agent_record_list.append(a.log_agent())
                if self.t0 is not None and a.desired_start_time >= self.t0:
                    self.n_valid_density_agents += 1
                    if self.n_valid_density_agents % 10 == 0:
                        self.my_logger.log('Valid agents that have completed their flight: '+str(self.n_valid_density_agents)+', at time '+str(time))
                agents_to_remove.append(a)
            if not self.ramping_up and self.t0 is None:
                if a.start_time < self.t_density:
                    found_old_agents = True
            else:
                found_old_agents = True
        self.agent_oldest_time = agent_oldest_time
        if not found_old_agents and self.t0 is None:
            self.t0 = time  # t0 time at which all agents in the simulation have been created at the target density
            self.my_logger.log('All agents in the simulation have been created at the target density, t0: '+str(self.t0))
        for a in agents_to_remove:
            #check if it can land on the destination
            if self.vertiport_db.loc[a.goal_id,'land_capacity']>0:
                self.vertiport_db.loc[a.goal_id,'land_capacity'] -= 1
                self.vertiport_db.loc[a.goal_id,'takeoff_capacity'] += 1
                self.remove_agent(a, time)
            else:
                a.flight_phase = 'hold'

    def terminate(self):
        # Go through all the conflicts and agents that are leftovers
        for agent in self.smart_agents+self.dumb_agents:
            if self.log_type != 'short' or agent.agent_logic != 'dumb':
                self.agent_record_list.append(agent.log_agent())
        for agent in self.waiting_agent:
            # The simulation timed out without being able to take-off
            if self.log_type != 'short' or agent.agent_logic != 'dumb':
                self.agent_record_list.append(agent.log_agent())
        for conflict in self.conflicts_dic.values():
            self.conflicts_list.append(conflict.bundle_conflict())
        if self.agent_record_list == []:
            print('The agent log failed to be added')
            print('The waiting list is ', self.waiting_agent)
        log_data = {'conflicts': self.conflicts_list,
                    'agents': self.agent_record_list,
                    'times': {'time_density_is_reached': self.t_density, 'time_all_started_after_t_density': self.t0}
                    }
        if self.log_density:
            log_data['density_map'] = self.density_map.density.tolist()
        return log_data

    def update_density(self):
        for agent_pos in self.active_agent_pos:
            self.density_map.add_pos(agent_pos)

    def get_protected_area(self):
        # Only works for one smart agent
        dic = None
        for agent in self.smart_agents:
            dic = {'center': agent.position, 'radius': agent.radius}
        return dic

    def add_random_agent(self, random_type='edge', start_time=0.0, density=0):
        if random_type == 'edge':
            self.total_number_of_agents += 1
            a = agent.Agent(self, self.min_distance, self.speed, start_time=start_time, id=self.total_number_of_agents)
        elif random_type == 'in map':
            # start in the middle, end at an edge
            start = [random.randint(0, self.size_map), random.randint(0, self.size_map)]
            edge = random.randint(0, 3)
            along_edge = random.randint(0, self.size_map)
            x = along_edge * (-(edge % 2) + 1) + (edge == 1) * self.size_map
            y = along_edge * (edge % 2) + (edge == 2) * self.size_map
            end = [x, y]
            self.total_number_of_agents += 1
            a = agent.Agent(self, self.min_distance, self.speed, start=np.asarray(start, dtype=np.float32),
                            end=np.asarray(end, dtype=np.float32), start_time=start_time, id=self.total_number_of_agents)
        if self.utm_on:
            flightplan = a.preflight(self.default_planning_time_step, density=density)
            self.centralized_manager.add_flight_plan(flightplan, a.id)
            #self.discrete_environment.addDynamicObstacle(flightplan)
        # TODO get rid of agent category
        self.dumb_agents.append(a)
        return a

    def add_intruder(self, start=None, end=None, delay=0, density=0):
        # Passing density along for logging purposes but not super clean
        self.total_number_of_agents += 1
        a = agent.Agent(self, self.min_distance, self.speed, start=start, end=end, start_time=delay, agent_logic='dumb', id=self.total_number_of_agents)
        if self.utm_on:
            flightplan = a.preflight(self.default_planning_time_step, density=density)
            self.centralized_manager.add_flight_plan(flightplan,a.id)
        self.dumb_agents.append(a)
        return a

    def add_phantom_agent(self, start, end):
        a = agent.Agent(self, self.min_distance, self.speed, start=start, end=end, agent_logic='dumb')
        self.phantom_agents.append(a)

    def add_reactive_agent(self, start=None, end=None, start_time=0.0, algo_type='MVP', flight_leg='initial'):
        self.total_number_of_agents += 1
        a = agent.Agent(self, self.min_distance, self.speed, start_time=start_time, start=start, end=end, agent_logic='reactive',
                        algo_type=algo_type, id=self.total_number_of_agents, sensing_radius=self.sensing_radius, flight_leg=flight_leg)
        # Don't do a preflight for reactive agents (not supported now)
        # Not a full preflight just ground delay if start position is not available
        self.waiting_agent.append(a)
        return a

    def add_strategic_agent(self, start=None, end=None, start_time=0.0, algo_type='A_star', density=0, flight_leg='inital'):
        # Strategic agents need a UTM to plan their flights
        if self.centralized_manager is None:
            print('Strategic agents need a centralized manager to be on to plan their flights')
            print('agent not added')
            return None
        else:
            self.total_number_of_agents += 1
            a = agent.Agent(self, self.min_distance, self.speed, start=start, end=end, start_time=start_time, agent_logic='strategic',
                            centralized_manager=self.centralized_manager, id=self.total_number_of_agents, sensing_radius=self.sensing_radius, flight_leg=flight_leg)
            # TODO tolerance issues
            # Profile the following function
            #
            # import cProfile, pstats, io
            # from pstats import SortKey
            # pr = cProfile.Profile()
            # pr.enable()

            flightplan = a.preflight(self.default_planning_time_step, algo_type=algo_type, density=density)
            #
            # pr.disable()
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())

            if flightplan is None:
                print('strategic agent failed to plan')
                self.agent_record_list.append(a.log_agent())
            else:
                self.centralized_manager.add_flight_plan(flightplan,a.id)
                self.smart_agents.append(a)
            # DEBUG don't add the agent to the list of obstacles
            # self.discrete_environment.addDynamicObstacle(flightplan)
            return a

    def remove_agent(self, a, time, density=0):
        if a.agent_logic != "dumb":
            self.smart_agents.remove(a)
        else:
            self.dumb_agents.remove(a)
        if self.utm_on:
            self.centralized_manager.terminate_flight_plan(a.id)
        i = self.active_agents.index(a)
        del self.active_agents[i]
        del self.active_agent_pos[i]
        if self.multiple_planning_agents:
            if self.simulation_type == 'reactive':
                self.add_reactive_agent(algo_type=self.algo_type, start_time=time)
            elif self.simulation_type == 'strategic':
                self.add_strategic_agent(algo_type=self.algo_type, start_time=time, density=density)
        else:
            if self.simulation_type == 'reactive':
                self.add_random_agent(random_type='edge', start_time=time)

    def add_agents(self, time):
        if self.multiple_planning_agents:
            if self.ramping_up:
                # Add agents from time to time
                # TODO fix the time interval
                time_interval=1570.0/self.desired_number_of_agents
                if time-self.agent_added_time>time_interval:
                    self.agent_added_time=time
                    if self.simulation_type=='reactive':
                        self.add_reactive_agent(algo_type=self.algo_type, start_time=time)
                    elif self.simulation_type == 'strategic':
                        self.add_strategic_agent(algo_type=self.algo_type,start_time=time)

    def build_kd_tree(self):
        if self.active_agent_pos == []:
            self.agents_kd_tree = None
        else:
            self.agents_kd_tree = KDTree(np.array(self.active_agent_pos))

    def get_neighbors(self,position,radius):
        # Will also return itself unless an exclusion id is specified
        if self.agents_kd_tree is not None:
            ind = self.agents_kd_tree.query_radius(position.reshape(1,-1),radius,return_distance=False)
            return np.array(self.active_agents)[ind[0]]
        else:
            return []

    def get_nearest_neighbors(self, position, k, radius):
        """Returns the k-closest neighbors within radius distance"""
        if self.agents_kd_tree is not None:
            k = min(k, len(self.active_agent_pos))
            distances, indices = self.agents_kd_tree.query(position.reshape(1, -1), k=k, return_distance=True)
            max_index = np.searchsorted(distances[0], radius, side='right')
            if max_index == 0:
                return []
            else:
                return np.array(self.active_agents)[indices[0][0:max_index]]
        else:
            return []


class Conflict:
    def __init__(self, time, agent_set):
        """ The agent_set is a frozenset"""
        # self.agents=agent_set  # Hashable (immutable)
        self.agent1, self.agent2 = agent_set
        self.start_time = time
        self.end_time = time
        self.min_separation = None
        self.min_h_separation = None
        self.min_z_separation = None

    def update(self, time):
        distance = np.linalg.norm(self.agent1.position - self.agent2.position)
        h_distance = distance
        z_distance = 0
        if self.min_separation is None or self.min_separation > distance:
            self.min_separation = distance
        if self.min_h_separation is None or self.min_h_separation > h_distance:
            self.min_h_separation = h_distance
        if self.min_z_separation is None or self.min_z_separation > z_distance:
            self.min_z_separation = z_distance
        self.end_time = time

    def bundle_conflict(self):
        """Returns the conflict attributes in a dictionary"""
        # vars(self) does not work because of the objects
        dic = {'agent1': self.agent1.id, 'agent2': self.agent2.id, 'start_time': self.start_time,
               'end_time': self.end_time,
               'min_separation': self.min_separation, 'min_h_separation': self.min_h_separation,
               'min_z_separation': self.min_z_separation}
        return dic


class DensityMap:
    def __init__(self, size_map, size_bin):
        self.size_map = size_map
        self.n_bins = math.ceil(self.size_map / size_bin)
        self.size_bin = self.size_map / self.n_bins
        self.density = np.zeros((self.n_bins, self.n_bins))

    def add_pos(self, position):
        i = min(int(position[0] / self.size_bin),
                self.n_bins - 1)  # truncate (include points on the edge in the last bin)
        j = min(int(position[1] / self.size_bin), self.n_bins - 1)
        self.density[i, j] += 1
