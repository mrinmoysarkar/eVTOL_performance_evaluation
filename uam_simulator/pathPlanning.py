import heapq
import math
import numpy as np
import gurobipy as grb
from gurobipy import GRB
from uam_simulator.my_utils import get_trajectory_circle_intersection


# For details about the implementation of a priority queue in python see # https://docs.python.org/2/library/heapq.html
class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.priority3=None

    def empty(self):
        return len(self.elements)==0

    def push(self, state, priority, priority2,priority3=None):
        # not checking for duplicates. According to redblob games, actually faster than implementing a smarter heapq
        # to break the ties between two cells with same f, favors the one with lowest h
        self.priority3=priority3
        if self.priority3 is not None:
            heapq.heappush(self.elements, (priority,priority2,priority3, state))
        else:
            heapq.heappush(self.elements, (priority, priority2, state))

    def pop(self):
        #return the lowest priority task
        while self.elements:
            if self.priority3 is not None:
                priority,priority2,priority3, state = heapq.heappop(self.elements)
            else:
                priority, priority2, state = heapq.heappop(self.elements)
            return state
        raise KeyError('pop from an empty priority queue')


class DecoupledApproach:
    def __init__(self,start,goal,start_time, speed, centralized_manager, tolerance):
        self.start=start
        self.goal=goal
        self.centralized_manager=centralized_manager
        self.start_time=start_time
        self.speed=speed
        self.tolerance = tolerance

    def heuristic(self,node_t):
        remaining_distance=self.end_node.i-node_t.node.i
        return remaining_distance

    def search(self):
        step=self.centralized_manager.minimum_separation_distance
        # TODO need to find a time step for whe
        my_line=Line(self.start,self.goal,step,self.start_time, self.speed)
        start_node=my_line.get_start_node()
        start_node_t = start_node.get_node_at_k(0)
        self.end_node = my_line.get_end_node()
        open_queue = PriorityQueue()
        open_queue.push(start_node_t, self.heuristic(start_node_t), self.heuristic(start_node_t),self.heuristic(start_node_t))
        open_cost_so_far = {}
        open_energy_so_far={}
        came_from = {}
        open_cost_so_far[start_node_t] = 0
        open_energy_so_far[start_node_t] = 0
        came_from[start_node_t] = None
        success = True
        if open_queue.empty():
            print('pathPlanning: A star failed to find a path, nothing in the open_queue')
            success = False
        while not open_queue.empty():
            current = open_queue.pop()
            # if current.node == self.end_node:
            #     break
            if current.node!=start_node and np.linalg.norm(current.node.pos-self.end_node.pos) <= self.tolerance:
                break
            for neighbor in my_line.get_neighbours(current):
                if neighbor.t==current.t:
                    print("this should not happen, line search")
                    print(my_line.get_neighbours(current))
                    print("neighbor k "+str(neighbor.k))
                    print("current k "+str(current.k))
                    k = my_line.convert_t2k(current.t)
                    print('k value '+str(k))
                on_the_ground = current.node==neighbor.node and current.node==start_node
                if on_the_ground or self.centralized_manager.is_path_available(current.t,current.node.pos,neighbor.node.pos, neighbor.t):
                    cost_to_go,energy_to_go=my_line.travel_cost(current, neighbor)
                    new_energy = open_energy_so_far[current] + energy_to_go
                    new_cost = open_cost_so_far[current] + cost_to_go
                    if neighbor not in open_cost_so_far or open_cost_so_far[neighbor] > new_cost or (open_cost_so_far[neighbor]==new_cost and open_energy_so_far[neighbor]>new_energy):
                        open_cost_so_far[neighbor] = new_cost
                        open_energy_so_far[neighbor] = new_energy
                        # TODO energy stuff not working
                        open_queue.push(neighbor, new_cost + self.heuristic(neighbor),new_energy+self.heuristic(neighbor), self.heuristic(neighbor) )
                        came_from[neighbor] = current
            if open_queue.empty():
                print('pathPlanning: A star failed to find a path, nothing in the queue')
                success = False
        plan = []
        times = []
        if success:
            previous = came_from[current]
            if previous is None:
                # Agent starts within tolerance of its goal
                print(self.start)
                print(self.goal)
                return True, [self.start,self.goal], []
            direction = current.node.pos-previous.node.pos
            velocity = self.speed*direction/np.linalg.norm(direction)
            t, pos = get_trajectory_circle_intersection(self.end_node.pos, 0.95*self.tolerance, previous.node.pos, velocity)
            plan.append(pos[0])
            times.append(previous.t+t[0])
            if times[0] < 0:
                print('pathPlanning issue in Decoupled Approach')
            # plan.append(current.node.pos)
            # times.append(current.t)
            parent = came_from[current]
            stop = False
            while parent is not None and (not stop):
                plan.append(parent.node.pos)
                times.append(parent.t)
                if parent.node.pos[0] == self.start[0] and parent.node.pos[1] == self.start[1]:
                    stop = True
                parent = came_from[parent]
            times.reverse()
            plan.reverse()
        return success, plan, times


class SIPP:
    def __init__(self,start,goal,start_time, speed, centralized_manager, tolerance):
        self.start = start
        self.goal = goal
        self.centralized_manager = centralized_manager
        self.tolerance = tolerance
        self.i_goal,self.j_goal=self.centralized_manager.get_coordinates(self.goal)
        self.start_time = start_time
        self.speed = speed

    def heuristic(self,state):
        # Need to use diagonal distance!
        d = self.centralized_manager.cell_length
        d_diag = 1.4142135623730951*d
        i_node,j_node=self.centralized_manager.get_coordinates(state.position)
        dx=abs(i_node-self.i_goal)
        dy=abs(j_node-self.j_goal)
        distance=d*(dx+dy)+(d_diag-2*d)*min(dx,dy)
        return distance/self.speed

    def search(self, debug=False):
        open_queue=PriorityQueue()
        open_cost_so_far = {}
        came_from = {}
        # Initialization
        successors = self.centralized_manager.get_successors(self.start, self.start_time, self.speed, on_the_ground=True)
        for successor in successors:
            key=(successor[0][0],successor[0][1],successor[1][0],successor[1][1])
            g = successor[2]
            open_cost_so_far[key] = 0
            start_state = SafeState(successor[0], successor[2])
            came_from[start_state] = None
            h=self.heuristic(start_state)
            open_queue.push(start_state,g+h,h)
        success=True
        count=0
        on_the_ground=False
        while not open_queue.empty():
            current = open_queue.pop()
            count += 1
            # TODO fix when goal is not on the grid
            # if current.position[0]==self.goal[0] and current.position[1]==self.goal[1]:
            #     break
            if np.linalg.norm(current.position-self.goal) <= self.tolerance and np.linalg.norm(current.position - self.start) != 0:
                break
            successors = self.centralized_manager.get_successors(current.position, current.time, self.speed)
            for successor in successors:
                key = (successor[0][0], successor[0][1], successor[1][0], successor[1][1])
                g = successor[2]
                if key not in open_cost_so_far or open_cost_so_far[key]>g:
                    open_cost_so_far[key] = g
                    new_state = SafeState(successor[0], successor[2])
                    came_from[new_state] = current
                    h = self.heuristic(new_state)
                    open_queue.push(new_state, g+h, h)
            if open_queue.empty():
                success=False
                print('pathPlanning: SIPP failed to find a path')
                print('start ' + str(self.start))
                print('goal '+ str(self.goal))
                print('came from '+str(len(came_from)))
                print('Successor start '+str(self.centralized_manager.get_successors(self.start, self.start_time, self.speed, on_the_ground=True)))
                print('Current position '+str(current.position)+', time '+str(current.time))
                print('Current successors ' + str(self.centralized_manager.get_successors(current.position, current.time, self.speed, debug=True)))
        plan=[]
        times=[]
        if success:
            direction=self.goal-current.position
            d = np.linalg.norm(direction)
            if d == self.tolerance:
                # Travel a bit inside to make sure it gets deleted
                extra = 0.05*self.tolerance*direction/d
                plan.append(current.position + extra)
                times.append(current.time+0.05*self.tolerance/self.speed)
            position=current.position
            plan.append(position)
            time=current.time
            times.append(time)
            parent=came_from[current]
            stop=False
            while parent is not None and (not stop):
                old_time=parent.time
                old_position = parent.position
                t_intermediate = time-np.linalg.norm(position-old_position)/self.speed
                # Floating point error
                if abs(t_intermediate-old_time)>1e-5:
                    # it means the agent must wait in place
                    plan.append(old_position)
                    times.append(t_intermediate)
                if parent.position[0] == self.start[0] and parent.position[1] == self.start[1]:
                    stop = True
                    if abs(t_intermediate - old_time) <= 1e-5:
                        plan.append(old_position)
                        times.append(old_time)
                else:
                    plan.append(old_position)
                    times.append(old_time)

                time=old_time
                position=old_position
                parent=came_from[parent]
            times.reverse()
            plan.reverse()
        return success, plan, times


class SafeState:
    def __init__(self,position,time):
        self.position=np.array(position)
        self.time=time

    def __lt__(self,other):
        return self.time<other.time


class Line:
    def __init__(self, start, end, step, start_time, velocity):
        """start numpy array [x,y], end numpy array [x,y], step: float desired distance between two steps """
        self.nodes={}
        self.start_time=start_time
        d = np.linalg.norm(end-start)
        self.n = int(math.ceil(d/step))
        if self.n == 0:
            print('n is zero in line, distance '+str(d)+', step '+str(step))
        self.spatial_step=d/self.n  # Adjusted so that each step is the same size
        self.t_step=self.spatial_step/velocity
        for i in range(0,self.n+1):
            pos=start + i * self.spatial_step * (end-start)/d
            self.nodes[i]=Node(self,pos,i)

    def get_neighbours(self ,my_node_t):
        current_time=my_node_t.t
        k=self.convert_t2k(current_time)
        i=my_node_t.node.i
        neighbours=[]
        neighbours.append(self.nodes[i].get_node_at_k(k + 1))
        if i>0:
            neighbours.append(self.nodes[i-1].get_node_at_k(k+1))
        if i<self.n+1:
            neighbours.append(self.nodes[i+1].get_node_at_k(k+1))
        return neighbours

    def travel_cost(self,node_a,node_b):
        # Must be consistent with heuristic
        if node_a.node == self.get_start_node() and node_b.node==self.get_start_node():
            energy=0
        else:
            energy=min(abs(node_a.node.i-node_b.node.i), 1)
        return min(abs(node_a.node.i-node_b.node.i),1), energy

    def get_start_node(self):
        return self.nodes[0]

    def get_end_node(self):
        return self.nodes[self.n]

    def convert_t2k(self, time):
        """Continuous time to discrete"""
        return round((time-self.start_time)/self.t_step)

    def convert_k2t(self, k):
        return self.start_time+self.t_step*k


class Node:
    def __init__(self,line,pos,i):
        self.line=line
        self.i=i
        self.pos=pos
        self.time_dic={}

    def get_node_at_k(self,k):
        t=self.line.convert_k2t(k)
        if k in self.time_dic:
            return self.time_dic[k]
        else:
            return Node_t(self,k,t)


class Node_t:
    def __init__(self,my_node,k,t):
        self.node=my_node
        self.t = t
        self.k = k
        self.node.time_dic[k]=self

    def __lt__(self,other):
        return self.node.pos[0] < other.node.pos[0]


###################### Local Method
class Local_VO:
    def __init__(self, start, goal, start_time, speed, centralized_manager, tolerance):
        self.start = start
        self.goal = goal
        self.centralized_manager = centralized_manager
        self.tolerance = tolerance
        self.start_time = start_time
        self.speed = speed
        self.sensing_radius = 5000
        self.minimum_separation_distance = centralized_manager.minimum_separation_distance
        self.dt = 10
        self.k_nearest_neighbors = 10

    def search(self, debug=False):
        success = False
        if debug:
            print('Starting debugging')
        while not success:
            # check if you can take-off (i.e. there are no vehicles forecast right above you). If cannot take-off, wait for dt and check again
            pos = np.copy(self.start)
            intruders_pos, intruders_vel = self.centralized_manager.get_forecast_intruders(pos, self.minimum_separation_distance, self.start_time)
            while len(intruders_pos) != 0:
                if debug:
                    print('adding ground delay')
                self.start_time += self.dt
                intruders_pos, intruders_vel = self.centralized_manager.get_forecast_intruders(pos, self.minimum_separation_distance, self.start_time)
            time = self.start_time
            planned_times = [time]
            planned_positions = [np.copy(pos)]
            direction_to_goal = self.goal-pos
            distance_to_goal = np.linalg.norm(direction_to_goal)
            no_solution = False
            while distance_to_goal > self.tolerance and not no_solution:
                positions, velocities = self.centralized_manager.get_forecast_intruders(pos, self.sensing_radius, time)
                desired_vel = min(self.speed, distance_to_goal / self.dt) * direction_to_goal / distance_to_goal
                if len(positions) != 0:
                    # only consider the k closest intruders (get forecast intruder returns a sorted list)
                    k_max = min(self.k_nearest_neighbors, len(positions))
                    model = self.setupMIQCP(positions[0:k_max], velocities[0:k_max], desired_vel, pos)
                    if model is None:
                        no_solution = True
                        print('there would be a conflict ')
                        break
                    model.optimize()
                    if model.status != GRB.Status.OPTIMAL:
                        if debug:
                            print('Error gurobi failed to find a solution')
                            print(model.status)
                        no_solution = True
                        break
                    vars = model.getVars()
                    opt_vel = np.array([vars[0].x, vars[1].x])
                    if debug:
                        print('Optimized velocity')
                        print(opt_vel)
                        print(np.linalg.norm(opt_vel))
                else:
                    # If there are no intruders nearby, no need to optimize
                    if debug:
                        print('no intruders')
                    opt_vel = desired_vel
                planned_positions.append(pos + opt_vel*self.dt)
                planned_times.append(time+self.dt)
                pos = pos+opt_vel*self.dt
                time = time + self.dt
                direction_to_goal = self.goal - pos
                distance_to_goal = np.linalg.norm(direction_to_goal)
            if no_solution:
                # Try again at a later time (put it in a loop or call search iteratively?)
                success = False
                self.start_time += self.dt
            else:
                success = True
        print("*********************************")
        print(planned_positions)
        print(planned_times)
        print("*********************************")
        return success, planned_positions, planned_times

    def setupMIQCP(self, intruders_pos, intruders_vel, desired_vel, ownship_pos):
        """ Intruders should be an array of agents """
        model = grb.Model('VO')
        model.addVar(lb=-self.speed, ub=self.speed, name='x')
        model.addVar(lb=-self.speed, ub=self.speed, name='y')
        model.addVars(2 * len(intruders_pos), vtype=GRB.BINARY)
        model.update()
        X = model.getVars()
        n_intruder = 0
        for i in range(0, len(intruders_pos)):
            constraints_or = self.get_VO(intruders_pos[i], intruders_vel[i], ownship_pos)
            if constraints_or[0] is None:
                return None
            n_constraint = 0
            for constraint in constraints_or:
                c = constraint(0, 0)
                a = constraint(1, 0) - c
                b = constraint(0, 1) - c
                # K must be arbitrarily large so that when the binary constraint is 1 the constraint is always respected
                # If K is chosen too large it creates issues for the solver. K is chosen just large enough.
                K = abs(a * self.speed) + abs(b * self.speed) + c
                model.addConstr(a * X[0] + b * X[1] - K * X[2 + 2 * n_intruder + n_constraint] <= -c)
                n_constraint += 1
            model.addConstr(X[2 + 2 * n_intruder] + X[2 + 2 * n_intruder + 1] <= 1)
            n_intruder += 1
        model.addConstr(X[0] * X[0] + X[1] * X[1] <= self.speed ** 2)
        model.setObjective((X[0] - desired_vel[0]) * (X[0] - desired_vel[0]) + (X[1] - desired_vel[1]) * (X[1] - desired_vel[1]), GRB.MINIMIZE)
        model.setParam("OutputFlag", 0)
        model.setParam("FeasibilityTol", 1e-9)
        model.update()
        return model

    def get_VO(self, intruder_pos, intruder_vel, ownship_pos):
        rel_pos = intruder_pos - ownship_pos
        d = np.linalg.norm(rel_pos)
        if self.minimum_separation_distance > d:
            # There is a loss of separation
            return None, None
        alpha = math.asin(self.minimum_separation_distance / d)  # VO cone half-angle (>=0)
        theta = math.atan2(rel_pos[1], rel_pos[0])
        vector1 = [math.cos(theta + alpha), math.sin(theta + alpha)]
        vector2 = [math.cos(theta - alpha), math.sin(theta - alpha)]
        # must be greater
        normal_1 = np.array([vector1[1], -vector1[0]])  # Rotated +90 degrees
        constraint1 = lambda x, y: np.dot((np.array([x, y]) - intruder_vel) + 0.1 * normal_1, normal_1)
        # must be smaller
        normal_2 = np.array([-vector2[1], vector2[0]])  # Rotated -90 degrees
        constraint2 = lambda x, y: np.dot((np.array([x, y]) - intruder_vel) + 0.1 * normal_2, normal_2)
        return constraint1, constraint2



