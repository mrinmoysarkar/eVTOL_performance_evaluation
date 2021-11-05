import math
import numpy as np
from uam_simulator import my_utils
from uam_simulator import pathPlanning
from uam_simulator import orca
import gurobipy as grb
from gurobipy import GRB
import time as python_time


class Flightplan:
    def __init__(self, t0, dt, positions, times=None):
        self.start_time = t0
        self.positions = positions
        self.time_step = dt
        self.times = None
        if times is not None:
            self.time_step = None
            self.times = np.array(times)
        else:
            self.times = np.array([self.start_time + i * self.time_step for i in range(0, len(self.positions))])
        if self.time_step is not None:
            self.end_time = self.start_time + (len(self.positions) - 1) * self.time_step
        else:
            self.end_time=self.times[-1]

    def get_planned_position_at(self, time, return_velocity=False, ignore_timed_out=False, debug=False):
        """ Interpolates between the flight points """
        if ignore_timed_out and time > self.end_time:
            if return_velocity:
                return None, None
            else:
                return None
        n = len(self.positions)
        if self.time_step is not None:
            idx_float = (float(time) - float(self.start_time)) / float(self.time_step)
            idx_low = min(math.floor(idx_float), n - 1)
            idx_high = min(math.ceil(idx_float), n - 1)
            if idx_low == idx_high:
                # Indices are equal because idx_float is an int
                if return_velocity:
                    if idx_high == n-1:
                        velocity = np.array([0, 0])
                    else:
                        velocity = (self.positions[idx_high+1]-self.positions[idx_high])/(self.times[idx_high+1]-self.times[idx_high])
                    return self.positions[idx_high], velocity
                else:
                    return self.positions[idx_high]
        else:
            if time > self.times[-1]:
                return np.copy(self.positions[-1])
            idx_high = np.searchsorted(self.times, time)
            idx_low = max(0, idx_high - 1)
            if self.times[idx_high] == time or idx_low == idx_high:
                if return_velocity:
                    if idx_high == n-1:
                        velocity = np.array([0, 0])
                    else:
                        velocity = (self.positions[idx_high+1]-self.positions[idx_high])/(self.times[idx_high+1]-self.times[idx_high])
                    return self.positions[idx_high], velocity
                else:
                    return self.positions[idx_high]
            idx_float = idx_low + (time - self.times[idx_low]) / (self.times[idx_high] - self.times[idx_low])

        pos_high = self.positions[idx_high]
        pos_low = self.positions[idx_low]  # if time is exactly integer then returns the exact pos

        if debug:
            print(idx_float)
            print(pos_high)
            print(pos_low)
        if return_velocity:
            return pos_low + (pos_high - pos_low) * (idx_float - idx_low), (pos_high-pos_low)/(self.times[idx_high]-self.times[idx_low])
        else:
            return pos_low + (pos_high - pos_low) * (idx_float - idx_low)

    def get_planned_trajectory_between(self, start_time, end_time, debug=False):
        """ Returns trajectory between start_time and end_time"""
        if (start_time - self.end_time) >= -1e-4 or (end_time - self.start_time) <= 1e-4:
            return None, None
        trajectory_end_time = min(end_time, self.end_time)
        trajectory_start_time = max(start_time, self.start_time)
        trajectory = []
        times = []
        if debug:
            print('time step is '+str(self.time_step))
            print('start_time is '+str(start_time))
            print('end_time '+str(end_time))
            print('positions '+str(self.positions))
            print('times '+str(self.times))
            print(start_time-self.end_time)
        if self.time_step is None:
            # self.times is sorted
            [start_index, end_index] = np.searchsorted(self.times, [trajectory_start_time, trajectory_end_time])
            temp = self.times[start_index]
            if abs(self.times[start_index]-trajectory_start_time) > 1e-4:
                # requires interpolation
                # Since we already now the index we could avoid a second call to search sorted
                trajectory.append(self.get_planned_position_at(trajectory_start_time))
                times.append(trajectory_start_time)
            for i in range(start_index, end_index):
                trajectory.append(self.positions[i])
                times.append(self.times[i])
            # trajectory_end_time <= times[end_index]
            if abs(self.times[end_index]-trajectory_end_time) > 1e-4:
                # requires interpolation
                trajectory.append(self.get_planned_position_at(trajectory_end_time))
                times.append(trajectory_end_time)
            else:
                trajectory.append(self.positions[end_index])
                times.append(trajectory_end_time)
        else:
            start_index_float = float((trajectory_start_time - self.start_time) / self.time_step)
            end_index_float = float((trajectory_end_time - self.start_time) / self.time_step)
            lower = math.ceil(start_index_float)
            upper = min(math.floor(end_index_float), len(self.positions) - 1)
            if lower != start_index_float:
                pos_0 = self.get_planned_position_at(start_time)
                trajectory.append(np.copy(pos_0))
                times.append(trajectory_start_time)
            for index in range(lower, upper + 1):
                trajectory.append(self.positions[index])
                # times.append(self.start_time+index*self.time_step)
                times.append(self.times[index])
            if upper != end_index_float:
                pos_end = self.get_planned_position_at(end_time)
                trajectory.append(pos_end)
                times.append(trajectory_end_time)
        return trajectory, times

    def get_end_time(self):
        if self.time_step is not None:
            return self.start_time + (len(self.positions) - 1) * self.time_step
        else:
            return self.times[-1]


class Agent:
    def __init__(self, env, radius, max_speed, start=None, end=None, start_time=0, agent_logic='dumb',
                 centralized_manager=None, algo_type=None, agent_dynamics=None, id=0, sensing_radius=10000,
                 flight_leg='initial'):
        self.id = id
        self.environment = env
        self.centralized_manager = centralized_manager
        self.agent_dynamics = agent_dynamics
        if agent_logic == 'dumb':
            protected_area = self.environment.get_protected_area()
        else:
            # All other agents can wait in place
            protected_area = None
        # Can't have random start and not random end (or vice versa)
        if start is None or end is None:
            self.start, self.goal, self.start_id, self.goal_id = self.environment.get_random_start_and_end(protected_area_start=protected_area)
            if np.linalg.norm(self.start - self.goal) < 10:
                # Play one more time
                # print('agent start and goal are close, redrawing at random')
                self.start, self.goal, self.start_id, self.goal_id = self.environment.get_random_start_and_end(protected_area_start=protected_area)
                if np.linalg.norm(self.start - self.goal) < 10:
                    print('unlikely, agent start and goal are still close')
        else:
            self.start = start
            self.goal = end
        self.position = np.copy(self.start)  # Passed by reference
        self.new_position = np.copy(self.start)
        self.radius = radius
        self.orientation = 0
        self.heading = np.degrees(np.arctan2(self.goal[1]-self.start[1],
                                             self.goal[0]-self.start[0]))
        if self.heading < 0:
            self.heading += 360
        self.minSpeed = 0.0
        self.maxSpeed = max_speed
        self.sensing_radius = sensing_radius
        self.desired_start_time = start_time
        self.start_time = start_time # actual start time if a ground delay is planned
        if np.linalg.norm(self.goal - self.start) == 0:
            print(agent_logic)
            print(start)
            print(end)
            print(np.linalg.norm(self.goal - self.start))
        self.velocity = self.maxSpeed * (self.goal - self.start) / (np.linalg.norm(self.goal - self.start))
        self.new_velocity = self.velocity
        self.trajectory = []
        self.trajectory_times = []
        self.collision_avoidance_time = []
        self.preflight_time = None
        self.flightPlan = None
        self.status = 'ok'
        self.agent_logic = agent_logic
        self.tolerance = self.environment.tolerance
        self.t_removed_from_sim=None
        if agent_logic == 'dumb':
            self.ownship = False
        else:
            self.ownship = True
        self.flight_status = 'initialized'
        self.algo_type = algo_type
        self.cumulative_density=0
        self.density=0
        self.n_steps=0
        self.flight_leg=flight_leg
        self.trajectory_4DT = []
        self.flight_phase = 'nothold'

    def get_predicted_end_time(self):
        if self.flightPlan is not None:
            return self.flightPlan.end_time
        else:
            print('Agent: in order to get the predicted end time a flight plan must exist')
            return self.start_time

    def compute_next_move(self, current_time, dt, debug=False, density=0):
        """ Store the next position in self.new_position. The position is updated when move is called """
        if self.flight_phase != 'hold':
            if self.agent_logic == 'dumb':
                self.new_position = self.compute_straight_move(self.position, self.goal, self.maxSpeed, dt)
                self.new_velocity = (self.new_position - self.position) / dt
            if self.agent_logic == 'reactive':
                self.cumulative_density += density
                self.n_steps += 1
                if self.algo_type is None:
                    self.algo_type = 'MVP'
                self.new_velocity = self.collision_avoidance(dt, algo_type=self.algo_type)
                self.new_velocity = self.velocity_update(self.new_velocity)
                self.new_position += self.new_velocity * dt
            if self.agent_logic == 'strategic':
                # Follow flight plan (without consideration for kinematic properties)
                self.new_position = self.flightPlan.get_planned_position_at(current_time + dt, debug=debug)
                self.new_velocity = (self.new_position - self.position) / dt
                if debug:
                    print('New position ' + str(self.new_position))
                    print('old position ' + str(self.position))
        
        if self.trajectory == []:
            self.trajectory.append(np.copy(self.position))
            self.trajectory_times.append(current_time)
            self.trajectory_4DT.append([self.position[0], self.position[1], 
                                        self.velocity[0], self.velocity[1], current_time])
        self.trajectory.append(np.copy(self.new_position))
        self.trajectory_times.append(current_time + dt)
        self.trajectory_4DT.append([self.new_position[0], self.new_position[1], 
                                    self.new_velocity[0], self.new_velocity[1], current_time + dt])
        self.flight_status = 'ongoing'

    def compute_straight_move(self, current_position, goal, speed, dt):
        orientation = math.atan2(goal[1] - current_position[1], goal[0] - current_position[0])
        d = np.linalg.norm(goal - current_position)
        max_step_length = min(speed * dt, d)  # slow down to arrive at the goal on the next time step
        return current_position + np.array([math.cos(orientation), math.sin(orientation)]) * max_step_length

    def move(self):
        # self.heading = np.degrees(np.arctan2(self.new_position[1]-self.position[1], self.new_position[0]-self.position[0]))
        self.position = np.copy(self.new_position)
        self.velocity = np.copy(self.new_velocity)
        
        return self.position

    def velocity_update(self, new_velocity):
        # Introduce kinematic constraints
        # For now just clamp the velocity and instantly change the orientation
        v = np.linalg.norm(new_velocity)
        v_clamped = my_utils.clamp(self.minSpeed, self.maxSpeed, v)
        if self.agent_dynamics is None:
            return new_velocity * v_clamped / v
        else:
            turn_angle = my_utils.get_angle(self.velocity, new_velocity)
            max_angle=30*math.pi/180
            if abs(turn_angle)>max_angle:
                vel = self.velocity * v_clamped / np.linalg.norm(self.velocity)
                theta=math.copysign(max_angle,turn_angle)
                return vel @ np.asarray([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            else:
                return new_velocity * v_clamped / v

    def preflight(self, dt, algo_type='Straight', density=0):
        # Given, the start/goals and published flight plans of other agents find a free path and publish it
        self.density = density
        if self.centralized_manager is None:
            print('agent.py preflight error, a centralized manager must exist')
        if algo_type == 'Straight':
            timer_start = python_time.time()
            plan = []
            plan.append(self.start)
            pos = np.copy(self.start)
            d = np.linalg.norm(self.goal - pos)
            # Larger time steps require larger tolerance
            # TODO tolerances are a bit of a mess
            while d > self.maxSpeed * dt:
                pos = self.compute_straight_move(pos, self.goal, self.maxSpeed, dt)
                d = np.linalg.norm(self.goal - pos)
                plan.append(pos)
            if d != 0:
                plan.append(self.goal)
            self.flightPlan = Flightplan(self.start_time, dt, plan)
            timer_end = python_time.time()
            self.preflight_time = timer_end - timer_start
            return self.flightPlan
        if algo_type == 'LocalVO':
            timer_start = python_time.time()
            local_planner = pathPlanning.Local_VO(self.start, self.goal, self.start_time, self.maxSpeed, self.centralized_manager, self.tolerance)
            success, plan, times = local_planner.search()
            if not success:
                self.flight_status = 'cancelled'
                return None
            self.start_time = times[0]
            ## Debug
            if len(times) < 2:
                print('the plan is too short')
                print('agent start '+ str(self.start))
                print('agent goal '+str(self.goal))
                print('agent plan pos '+str(plan))
                print('agent plan times ' + str(times))
            self.flightPlan = Flightplan(times[0], times[1] - times[0], plan, times=np.array(times))
            timer_end = python_time.time()
            self.preflight_time = timer_end - timer_start
            return self.flightPlan
        if algo_type == 'Decoupled':
            timer_start = python_time.time()
            decoupled_planner = pathPlanning.DecoupledApproach(self.start, self.goal, self.start_time, self.maxSpeed,
                                                               self.centralized_manager, self.tolerance)
            success, plan, times = decoupled_planner.search()
            if not success:
                self.flight_status = 'cancelled'
                return None
            self.start_time = times[0]
            self.flightPlan = Flightplan(times[0], times[1] - times[0], plan, times=np.array(times))
            timer_end = python_time.time()
            self.preflight_time = timer_end - timer_start
            return self.flightPlan
        if algo_type == 'SIPP':
            timer_start = python_time.time()
            sipp_planner = pathPlanning.SIPP(self.start, self.goal, self.start_time, self.maxSpeed,
                                             self.centralized_manager, self.tolerance)
            success, plan, times = sipp_planner.search()
            if not success:
                self.flight_status = 'cancelled'
                return None
            self.start_time = times[0]
            self.flightPlan = Flightplan(times[0], times[1] - times[0], plan, times=np.array(times))
            timer_end = python_time.time()
            self.preflight_time = timer_end - timer_start
            return self.flightPlan
        if algo_type == 'A_star_8':
            timer_start = python_time.time()
            astar_planner = pathPlanning.AStar_8grid(self.start, self.goal, self.start_time, self.maxSpeed,
                                                     self.centralized_manager)
            success, plan, times = astar_planner.search()
            if not success:
                self.flight_status = 'cancelled'
                timer_end = python_time.time()
                self.preflight_time = timer_end - timer_start
                return None
            self.start_time = times[0]
            self.flightPlan = Flightplan(times[0], times[1] - times[0], plan, times=np.array(times))
            timer_end = python_time.time()
            self.preflight_time = timer_end - timer_start
            return self.flightPlan
        else:
            print('The algo type ' + algo_type + ' is not implemented')

    def can_safely_take_off(self, t):
        if self.algo_type == 'straight':
            return True
        neighbors = self.environment.get_neighbors(self.position, self.radius)
        for vehicle in neighbors:
            if t >= vehicle.start_time and vehicle.id != self.id:
                # if np.linalg.norm(self.position - vehicle.position) <= self.radius:
                self.flight_status = 'waiting'
                return False
        # check the capacity of the vertiport
        if self.environment.vertiport_db['takeoff_capacity'][self.start_id] > 0:
            self.environment.vertiport_db.loc[self.start_id, 'takeoff_capacity'] -= 1
            self.environment.vertiport_db.loc[self.start_id, 'land_capacity'] += 1
        else:
            # print(self.environment.vertiport_db.takeoff_capacity)
            # print(self.environment.vertiport_db.land_capacity)
            return False
        
        self.start_time = t
        return True

    def collision_avoidance(self, dt, algo_type='MVP'):
        # Given current position, next flight plan goal and surrounding vehicles decide where to go
        # Based on Hoekstra Bluesky simulator
        # The returned velocity might not be feasible
        if algo_type == 'MVP_Bluesky':
            timer_start = python_time.time()
            neighbors = self.get_neighbors()
            velocity_change = np.asarray([0.0, 0.0])
            direction = self.goal - self.position
            d = np.linalg.norm(direction)
            desired_velocity = min(self.maxSpeed, d / dt) * direction / d
            safety_factor = 1.10  # 10% safety factor (as in The effects of Swarming on a Voltage Potential-Based Conflict Resolution Algorithm, T. Maas)
            # if d<=self.radius:
            #     dV=0
            # else:
            for neighbor in neighbors:
                # Find Time of Closest Approach
                delta_pos = self.position - neighbor.position
                dist=np.linalg.norm(delta_pos)
                if dist == 0:
                    continue
                delta_vel = desired_velocity - neighbor.velocity
                if np.linalg.norm(delta_vel)==0:
                    t_cpa=0
                else:
                    t_cpa=-np.dot(delta_pos, delta_vel) / np.dot(delta_vel, delta_vel)
                dcpa = delta_pos+delta_vel*t_cpa
                dabsH = np.linalg.norm(dcpa)
                # If there is a conflict
                if dabsH < self.radius:
                    # If head-on conflict
                    if dabsH<=10:
                        dabsH=10
                        dcpa[0] = delta_pos[1] / dist * dabsH
                        dcpa[1] = -delta_pos[0] / dist * dabsH
                    if self.radius*safety_factor < dist:
                        erratum = np.cos(np.arcsin((self.radius*safety_factor) / dist) - np.arcsin(dabsH / dist))
                        dV =(((self.radius*safety_factor) / erratum - dabsH) * dcpa)/(abs(t_cpa)*dabsH)
                    else:
                        # If already moving away from conflict (tcpa is negative) then just keep going
                        if t_cpa<=0:
                            dV = 0
                        else:
                            dV =(self.radius*safety_factor - dabsH)*dcpa/(abs(t_cpa)*dabsH)
                    velocity_change += dV
            timer_end = python_time.time()
            self.collision_avoidance_time.append(timer_end - timer_start)
            return desired_velocity + velocity_change
        elif algo_type == 'VO':
            timer_start = python_time.time()
            intruders = self.get_neighbors()
            d = np.linalg.norm(self.goal - self.position)
            speed = min(d / dt, self.maxSpeed)
            if d == 0:
                print('VO, this should not happen')
                print('distance to goal is 0')
            desired_velocity = (self.goal - self.position) * speed / d
            model = setupMIQCP(intruders, desired_velocity, self)
            model.optimize()
            if model.status != GRB.Status.OPTIMAL:
                print('Error gurobi failed to find a solution')
                print(model.status)
            vars = model.getVars()
            if intruders != []:
                # plotter([-1000,1000],[-1000,1000],100,[get_VO(intruders[0],self)],chosen_v=np.array([vars[0].x,vars[1].x]))
                pass
            timer_end = python_time.time()
            self.collision_avoidance_time.append(timer_end - timer_start)
            return np.array([vars[0].x, vars[1].x])
        elif algo_type == 'ORCA':
            timer_start = python_time.time()
            reactive_solver = orca.ORCA()
            vel=reactive_solver.compute_new_velocity(self, dt)
            timer_end = python_time.time()
            self.collision_avoidance_time.append(timer_end - timer_start)
            return vel
        elif algo_type == 'straight':
            timer_start = python_time.time()
            d = np.linalg.norm(self.goal - self.position)
            speed = min(d / dt, self.maxSpeed)
            desired_velocity = (self.goal - self.position) * speed / d
            timer_end = python_time.time()
            self.collision_avoidance_time.append(timer_end - timer_start)
            return desired_velocity
        else:
            print(algo_type+' not implemented ')

    def get_neighbors(self):
        neighbors = self.environment.get_neighbors(self.position, self.sensing_radius)
        if neighbors == []:
            return []
        else:
            return neighbors[neighbors != self]

    def get_nearest_neighbors(self, k, max_radius):
        # Will return itself so query one more neighbor
        neighbors = self.environment.get_nearest_neighbors(self.position, k+1, max_radius)
        if neighbors == []:
            return []
        else:
            return neighbors[neighbors != self]

    def finish_flight(self, t, goal_pos=None, t_removed_from_sim=None):
        self.flight_status = 'finished'
        self.arrival_time = t
        self.t_removed_from_sim=t_removed_from_sim
        if goal_pos is not None:
            self.trajectory.append(np.copy(goal_pos))
            self.trajectory_times.append(t)
            self.trajectory_4DT.append([goal_pos[0], goal_pos[1], 
                                        0, 0, self.arrival_time])

    def log_agent(self):
        agent_log = {'flight_status': self.flight_status,
                     'agent_type': self.agent_logic,
                     'desired_time_of_departure': self.desired_start_time,
                     'agent_id':self.id}
        if self.flight_status== 'finished' or self.flight_status == 'ongoing':
            agent_log['actual_time_of_departure'] = self.start_time
            if self.flight_status == 'finished':
                ideal_length = float(np.linalg.norm(self.goal - self.start))
                actual_length = 0
                if self.trajectory == []:
                    print('agent, empty trajectory ')  # happens if start and goal are really close
                    print(self.start)
                    print(self.goal)
                pos_0 = self.trajectory[0]
                for pos in self.trajectory:
                    d = np.linalg.norm(pos - pos_0)
                    actual_length += d
                    pos_0 = np.copy(pos)
                direction = self.goal - self.start
                heading = math.atan2(direction[1], direction[0])
                if self.agent_logic == 'reactive':
                    self.density = self.cumulative_density/self.n_steps - 1
                agent_log['flight_status']= self.flight_status
                agent_log['agent_type']= self.agent_logic
                agent_log['length_ideal']= ideal_length
                agent_log['actual_length']= actual_length
                agent_log['ideal_time_of_arrival']= self.desired_start_time+ideal_length / self.maxSpeed
                agent_log['actual_time_of_arrival']= self.arrival_time
                agent_log['4DT_Trajectory'] = self.trajectory_4DT
                if self.t_removed_from_sim is not None:
                    agent_log['time_removed_from_sim']=self.t_removed_from_sim
                agent_log['heading']= heading
                agent_log['density']= self.density
                if self.agent_logic == 'strategic':
                    agent_log['time_to_preflight'] = self.preflight_time
                elif self.agent_logic == 'reactive':
                    agent_log['average_time_to_plan_avoidance'] = sum(self.collision_avoidance_time) / len(self.collision_avoidance_time)
                    agent_log['total_planning_time'] = sum(self.collision_avoidance_time)
        return agent_log


def get_VO(intruder_agent, ownship_agent):
    if intruder_agent == ownship_agent:
        print('get_VO this should not happen intruder and ownship are the same')
    rel_pos = intruder_agent.position - ownship_agent.position
    d = np.linalg.norm(rel_pos)
    if d == 0:
        print('the distance between the two agents is 0')
    if ownship_agent.radius > d:
        print('there is an intruder in the protected radius')
        print(ownship_agent.position)
        print(intruder_agent.position)
    alpha = math.asin(ownship_agent.radius / d)  # VO cone half-angle (>=0)
    theta = math.atan2(rel_pos[1], rel_pos[0])
    vector1 = [math.cos(theta + alpha), math.sin(theta + alpha)]
    vector2 = [math.cos(theta - alpha), math.sin(theta - alpha)]
    # must be greater
    normal_1 = np.array([vector1[1], -vector1[0]])  # Rotated +90 degrees
    constraint1 = lambda x, y: np.dot((np.array([x, y]) - intruder_agent.velocity) + 0.1 * normal_1, normal_1)
    # must be smaller
    normal_2 = np.array([-vector2[1], vector2[0]])  # Rotated -90 degrees
    constraint2 = lambda x, y: np.dot((np.array([x, y]) - intruder_agent.velocity) + 0.1 * normal_2, normal_2)
    return constraint1, constraint2


def setupMIQCP(intruders, desired_vel, ownship_agent):
    """ Intruders should be an array of agents """
    model = grb.Model('VO')
    max_vel = ownship_agent.maxSpeed
    model.addVar(lb=-max_vel, ub=max_vel, name='x')
    model.addVar(lb=-max_vel, ub=max_vel, name='y')
    model.addVars(2 * len(intruders), vtype=GRB.BINARY)
    model.update()
    X = model.getVars()
    n_intruder = 0
    for intruder in intruders:
        constraints_or = get_VO(intruder, ownship_agent)
        n_constraint = 0
        for constraint in constraints_or:
            c = constraint(0, 0)
            a = constraint(1, 0) - c
            b = constraint(0, 1) - c
            # K must be arbitrarily large so that when the binary constraint is 1 the constraint is always respected
            K = abs(a * max_vel) + abs(b * max_vel) + c
            model.addConstr(a * X[0] + b * X[1] - K * X[2 + 2 * n_intruder + n_constraint] <= -c)
            n_constraint += 1
        model.addConstr(X[2 + 2 * n_intruder] + X[2 + 2 * n_intruder + 1] <= 1)
        n_intruder += 1
    model.addConstr(X[0] * X[0] + X[1] * X[1] <= max_vel ** 2)
    model.setObjective(
        (X[0] - desired_vel[0]) * (X[0] - desired_vel[0]) + (X[1] - desired_vel[1]) * (X[1] - desired_vel[1]),
        GRB.MINIMIZE)
    model.setParam("OutputFlag", 0)
    model.setParam("FeasibilityTol", 1e-9)
    model.update()
    return model