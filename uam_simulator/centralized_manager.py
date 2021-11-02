import numpy as np
import math
import copy
from uam_simulator.my_utils import clamp


class CentralizedManager:
    def __init__(self, min_separation):
        self.flight_plans={}
        self.minimum_separation_distance = min_separation

    def is_path_available(self, ownship_start_time, start_point, end_point, end_time):
        # How should collision avoidance be implemented ?
        # Naive approach
        # Iterate over all flight plans, find all flight segments that overlap in time with the input time
        # Compute for each of those segments the closest time of approach
        # Need to intersect time between time_start and time_end and get the resulting trajectory
        ownship_velocity = (end_point-start_point)/(end_time - ownship_start_time)
        for flight_plan in self.flight_plans.values():
            trajectory, times = flight_plan.get_planned_trajectory_between(ownship_start_time, end_time)
            if trajectory is not None:
                n=len(trajectory)
                intruder_segment_start_pos = np.copy(trajectory[0])
                intruder_segment_start_time = times[0]
                analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity
                for i in range(1,n):
                    intruder_segment_end_pos= trajectory[i]
                    intruder_segment_end_time = times[i]
                    if intruder_segment_end_time == intruder_segment_start_time:
                        print('trajectory '+str(trajectory))
                        print('times '+str(times))
                        print('ownship start time '+str(ownship_start_time))
                        print('end time '+str(end_time))
                    intruder_velocity = (intruder_segment_end_pos - intruder_segment_start_pos) / (intruder_segment_end_time - intruder_segment_start_time)
                    intruder_pos = intruder_segment_start_pos + (analysis_start_time - intruder_segment_start_time) * intruder_velocity
                    # Let's compute the closest time of approach
                    delta_v = intruder_velocity - ownship_velocity
                    delta_v_squared = np.dot(delta_v, delta_v)
                    delta_p = intruder_pos - ownship_pos
                    if delta_v_squared != 0:
                        t_cpa = analysis_start_time - np.dot(delta_v, delta_p) / delta_v_squared
                        # Let's find the time of closest approach on the segment
                        clamped_t_cpa=clamp(analysis_start_time,intruder_segment_end_time,t_cpa)-analysis_start_time  # between 0 and the end of the segment (t_elapsed)
                    else:
                        clamped_t_cpa = 0
                    closest = np.linalg.norm(intruder_pos+intruder_velocity*clamped_t_cpa-ownship_pos-ownship_velocity*clamped_t_cpa)
                    if closest < self.minimum_separation_distance:
                        return False
                    intruder_segment_start_time = intruder_segment_end_time
                    intruder_segment_start_pos = intruder_segment_end_pos
                    analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                    ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity
        return True

    def add_flight_plan(self,flight_plan, id):
        self.flight_plans[id]=flight_plan

    def terminate_flight_plan(self,id):
        if id in self.flight_plans:
            del self.flight_plans[id]
        else:
            print('Weird! the id should be in the flight plan')

    def validate_flight_plan(self, flight_plan):
        return True


class VOCentralizedManager:
    """Centralized manager for local path planning"""
    def __init__(self, min_separation):
        self.flight_plans = {}
        self.minimum_separation_distance = min_separation

    def is_path_available(self, ownship_start_time, start_point, end_point, end_time):
        ownship_velocity = (end_point - start_point) / (end_time - ownship_start_time)
        for flight_plan in self.flight_plans.values():
            trajectory, times = flight_plan.get_planned_trajectory_between(ownship_start_time, end_time)
            if trajectory is not None:
                n = len(trajectory)
                intruder_segment_start_pos = np.copy(trajectory[0])
                intruder_segment_start_time = times[0]
                analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity
                for i in range(1, n):
                    intruder_segment_end_pos = trajectory[i]
                    intruder_segment_end_time = times[i]
                    if intruder_segment_end_time == intruder_segment_start_time:
                        print('trajectory ' + str(trajectory))
                        print('times ' + str(times))
                        print('ownship start time ' + str(ownship_start_time))
                        print('end time ' + str(end_time))
                    intruder_velocity = (intruder_segment_end_pos - intruder_segment_start_pos) / (intruder_segment_end_time - intruder_segment_start_time)
                    intruder_pos = intruder_segment_start_pos + (analysis_start_time - intruder_segment_start_time) * intruder_velocity
                    # Let's compute the closest time of approach
                    delta_v = intruder_velocity - ownship_velocity
                    delta_v_squared = np.dot(delta_v, delta_v)
                    delta_p = intruder_pos - ownship_pos
                    if delta_v_squared != 0:
                        t_cpa = analysis_start_time - np.dot(delta_v, delta_p) / delta_v_squared
                        # Let's find the time of closest approach on the segment
                        clamped_t_cpa = clamp(analysis_start_time, intruder_segment_end_time, t_cpa) - analysis_start_time  # between 0 and the end of the segment (t_elapsed)
                    else:
                        clamped_t_cpa = 0
                    closest = np.linalg.norm(intruder_pos + intruder_velocity * clamped_t_cpa - ownship_pos - ownship_velocity * clamped_t_cpa)
                    if closest < self.minimum_separation_distance:
                        return False
                    intruder_segment_start_time = intruder_segment_end_time
                    intruder_segment_start_pos = intruder_segment_end_pos
                    analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                    ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity
        return True

    def add_flight_plan(self, flight_plan, id):
        self.flight_plans[id]=flight_plan

    def terminate_flight_plan(self, id):
        del self.flight_plans[id]

    def get_forecast_intruders(self, pos, radius, time,debug=False):
        distances = []
        positions = []
        velocities = []
        if debug:
            if time>=1060 and time <= 1080:
                pos_i, vel_i = self.flight_plans[11].get_planned_position_at(time, return_velocity=True, ignore_timed_out=True, debug=debug)
                print('Get planned pos at ')
                print(type(pos_i))
                print(pos_i)
                print(vel_i)
                print(pos)
        for flight_plan in self.flight_plans.values():
            pos_i, vel_i = flight_plan.get_planned_position_at(time, return_velocity=True, ignore_timed_out=True, debug=debug)
            if pos_i is not None:
                d = np.linalg.norm(pos_i - pos)
                if d < radius:
                    i = np.searchsorted(distances, d)
                    distances.insert(i, d)
                    positions.insert(i, pos_i)
                    velocities.insert(i, vel_i)
        return positions, velocities


class SIPPCentralizedManager:
    """Centralized manager implementation for safe-interval path planning"""
    def __init__(self, min_separation, size_area):
        self.flight_plans={}
        self.minimum_separation_distance = min_separation
        self.size_area=size_area
        self.n_grid=math.ceil(self.size_area/self.minimum_separation_distance)
        self.cell_length = self.size_area/self.n_grid
        self.n_grid+=1
        self.occupancy_grid = np.empty((self.n_grid,self.n_grid),dtype='object')
        for i in range(0,self.n_grid):
            for j in range(0,self.n_grid):
                self.occupancy_grid[i,j]=[[0,float('inf')]]

    def get_neighbours(self,position):
        neighbours=[]
        i_array=[]
        j_array=[]
        i,j=self.get_coordinates(position)
        i_array.append(i)
        j_array.append(j)
        if i>0:
            i_array.append(i-1)
        if i<self.n_grid-1:
            i_array.append(i+1)
        if j>0:
            j_array.append(j-1)
        if j<self.n_grid-1:
            j_array.append(j+1)
        for i_index in i_array:
            for j_index in j_array:
                if i_index!=i or j_index!=j:
                    neighbours.append([i_index,j_index])
        return neighbours

    def get_successors(self, position, time, speed, on_the_ground=False, debug=False):
        successors = []
        if on_the_ground:
            end_t = float('inf')
            start_t = time
            i, j = self.get_coordinates(position)
            for interval in self.occupancy_grid[i, j]:
                if interval[0] < end_t and interval[1] > start_t:
                    intersection=[max(interval[0],start_t),min(interval[1],end_t)]
                    first_arrival_time=intersection[0]
                    successors.append([position, interval, first_arrival_time])
        else:
            neighbours = self.get_neighbours(position)
            origin = np.copy(position)
            interval_departure = self.get_interval(position, time)
            if interval_departure is None:
                print("There is an issue get_successors was called but at the position and time specified there is no free interval (i.e. it's already occupied)")
                print('This should not have happened as the position should have been unreachable')
                print(position)
                print(time)
                i, j = self.get_coordinates(position)
                intervals = self.occupancy_grid[i, j]
                print(i)
                print(j)
                print(intervals)
                return successors
            if debug:
                print('time '+str(time))
                print('Departure interval : '+str(interval_departure))
            for neighbour in neighbours:
                destination = self.get_position(neighbour[0], neighbour[1])
                travel_time = np.linalg.norm(destination-origin)/speed
                start_t = time+travel_time
                end_t = interval_departure[1]+travel_time
                for interval in self.occupancy_grid[neighbour[0],neighbour[1]]:
                    if interval[0] < end_t and interval[1] > start_t:
                        intersection = [max(interval[0],start_t), min(interval[1], end_t)]
                        first_arrival_time = self.get_first_arrival_time(speed, origin, destination, intersection[0], intersection[1])
                        if debug:
                            print('first arrival time: '+str(first_arrival_time))
                        if first_arrival_time is not None:
                            successors.append([destination, interval, first_arrival_time])
        return successors

    def update_interval(self, interval, l,c):
        """ interval is an occupied interval, the function updates the list of free intervals when this function is
        called"""
        start_occupied = interval[0]
        end_occupied = interval[1]
        keep_iterating = True
        i = 0
        free_intervals = self.occupancy_grid[l,c]
        n_intervals = len(free_intervals)
        while keep_iterating:
            free_interval = free_intervals[i]
            start = free_interval[0]
            end = free_interval[1]
            # If the intervals overlap
            if end_occupied > start and start_occupied < end:
                # the occupied interval is entirely included in the free interval
                if start_occupied > start and end_occupied < end:
                    free_intervals.pop(i)
                    free_intervals.insert(i, [start, start_occupied])
                    free_intervals.insert(i + 1, [end_occupied, end])
                    keep_iterating = False
                # the occupied interval entirely covers the free interval
                elif start_occupied <= start and end_occupied >= end:
                    free_intervals.pop(i)
                    n_intervals -= 1
                # occupied interval at the start of free interval
                elif start_occupied <= start:
                    free_interval[0] = end_occupied
                    i += 1
                elif end_occupied >= end:
                    free_interval[1] = start_occupied
                    i += 1
            else:
                i += 1
            if i >= n_intervals:
                keep_iterating = False

    def get_first_arrival_time(self, v, origin,destination, start_time, end_time, on_the_ground=False, debug=False):
        # Start time and end time define the interval during which the agent can arrive at destination
        # is_path_available, returns the info of the intruder agent that is a problem if the ownship was to depart at departure_time
        # If the aircraft was delayed more than max_delay it would arrive outside of the interval or depart outside of
        # its origin interval
        max_delay = end_time-start_time
        travel_time = np.linalg.norm(origin-destination)/v
        earliest_arrival_time = start_time
        departure_time = earliest_arrival_time - travel_time
        available, intruder_trajectory = self.is_path_available(departure_time, origin, destination, earliest_arrival_time, debug=debug)
        count = 0
        P_o = origin
        V_o = v * (destination - origin) / np.linalg.norm(destination - origin)
        while earliest_arrival_time < end_time and not available:
            count += 1
            # Set position of everything at departure time
            V_i =(intruder_trajectory[0][1]-intruder_trajectory[0][0])/(intruder_trajectory[1][1]-intruder_trajectory[1][0])
            delta_t = intruder_trajectory[1][0]-departure_time
            P_i = intruder_trajectory[0][0]- delta_t * V_i
            # There is a conflict between the ownship and the intruder on the segment that was returned.
            # Note that when computing the required delay we are assuming the intruder started at P_i which is not true
            delay = self.get_time_to_leave(P_o, V_o, P_i, V_i, max_delay, on_the_ground=on_the_ground)
            if debug:
                print('''''''')
                print('departure time ' + str(departure_time))
                print('earliest arrival time '+str(earliest_arrival_time))
                print('intruder trajectory '+str(intruder_trajectory))
                print('delay '+str(delay))
                print('P_0' + str(P_o))
                print('V_0' + str(V_o))
                print('P_i' + str(P_i))
                print('V_i' + str(V_i))
            # Delay should be None if staying in place will result in a collision or if the delay will be more than
            # the max delay
            # Unless you are at the starting point (on_the_ground) in which case staying in place does not result in a
            # collision
            # TODO check if intruder is turning after looking for delay might lead to problem
            if delay is None:
                return None
            if departure_time + delay < intruder_trajectory[1][1]:
                # if departure_time+delay<intruder_trajectory[1][0]:
                #     print('departure time '+str(departure_time)+', delay '+str(delay)+', intruder departure time '+str(intruder_trajectory[1][0]))
                #     print('Centralized manager, get_first_arrival time: delay too small, this is unexpected and will result in issues')
                if debug:
                    print('found valid delay on intruder time')
                earliest_arrival_time = earliest_arrival_time+delay
            else:
                # The intruder turns before getting away
                # departure_time is at least the time at the end of the intruder trajectory
                earliest_arrival_time=intruder_trajectory[1][1]+travel_time
            # Is the delay enough to avoid the intruder or are there other intruders?
            if count > 100:
                count = 0
                debug = True
                print('found the culprit infinite loop')
                print('earliest time '+str(earliest_arrival_time))
                print('Maximum delay is '+str(max_delay))
                print('latest arrival time is '+str(max_delay+start_time))
                print('delay '+str(delay))
                print('The agent wants to go from '+str(origin)+' to '+str(destination))
                print('The agent is on the ground '+str(on_the_ground))
                print('It is conflicting with the flight plan '+str(intruder_trajectory))
                delay = self.get_time_to_leave(P_o, V_o, P_i, V_i, max_delay, on_the_ground=on_the_ground, debug=debug)
                print('P_0' + str(P_o))
                print('V_0' + str(V_o))
                print('P_i' + str(P_i))
                print('V_i' + str(V_i))

            departure_time = earliest_arrival_time - travel_time
            available, intruder_trajectory = self.is_path_available(departure_time, origin, destination, earliest_arrival_time, debug=debug)
        if available:
            return earliest_arrival_time
        else:
            return None  # No delay solved the issue

    def get_time_to_leave(self, P_o, V_o, P_i, V_i, max_time, on_the_ground=False, debug=False):
        # returns None if a conflict is unavoidable between time=0 and max_time
        # returns the value of a delay that allows to avoid the conflict
        # returns max_time
        delta_V = V_i - V_o
        delta_P = P_i - P_o
        delta_V_squared = np.dot(delta_V, delta_V)
        # Initial distance between ownship and intruder
        d_initial = np.linalg.norm(P_o - P_i)
        # Time of closest approach if the ownship does not move
        V_i_squared=np.dot(V_i, V_i)
        # To avoid numerical issues, make the avoidance area slightly bigger
        k_factor=1.0001
        if V_i_squared==0:
            # If the intruder is not moving there's no point in having a delay
            # This function is only called if there is a conflict on the time segment
            # Wait in place until the end of the segment
            t_cpa_stop = 0
            d_cpa_stop = d_initial
            if d_initial < self.minimum_separation_distance:
                return None
            else:
                return max_time
        else:
            t_cpa_stop = - np.dot(- V_i, P_o - P_i) / np.dot(V_i, V_i)
            d_cpa_stop = np.linalg.norm(P_i + t_cpa_stop * V_i - P_o)
        # Time of closest approach on the segment if the ownship does not move
        clamped_t_cpa_stop = clamp(0, max_time, t_cpa_stop)
        # Closest distance between the two agents on the segment if the ownship does not move (greater than d_cpa_stop)
        closest_distance_stop = np.linalg.norm(P_i + clamped_t_cpa_stop * V_i - P_o)
        if delta_V_squared == 0:
            a = -1
            Delta = -1
        else:
            # Time of closest approach if the ownship moves without delay
            t_cpa_no_delay = -np.dot(delta_V, delta_P) / delta_V_squared
            d_cpa_no_delay = np.linalg.norm(delta_P + t_cpa_no_delay * delta_V)
            Gamma = delta_P - delta_V * np.dot(delta_P, delta_V / delta_V_squared)
            Alpha = V_o - delta_V * (np.dot(delta_V, V_o) / delta_V_squared)
            c = np.dot(Gamma, Gamma) - (k_factor * self.minimum_separation_distance) ** 2
            b = 2 * np.dot(Alpha, Gamma)
            a = np.dot(Alpha, Alpha)
            Delta = b ** 2 - 4 * a * c
        if on_the_ground:
            if Delta >= 0 and a != 0:
                delay1 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                delay2 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if delay1 >= 0:
                    t_cpa_with_delay = -np.dot(delta_P + V_o * delay1, delta_V) / delta_V_squared
                    if delay1 <= t_cpa_with_delay:
                        return min(math.ceil(delay1 * 10) / 10, max_time)
                    else:
                        print('delay 1 greater than the time of closest approach found, no solution')
                        return None
                else:
                    if delay2 < 0:
                        print('This should not happen if there is a conflict, delay2<0')
                    else:
                        t_cpa_with_delay = -np.dot(delta_P + V_o * delay2, delta_V) / delta_V_squared
                        if delay2 < t_cpa_with_delay:
                            return min(math.ceil(delay2 * 10) / 10, max_time)
                        else:
                            # print('delay 2 greater than the time of closest approach found, no solution')
                            # print('P_o='+str(P_o))
                            # print('V_o=' + str(V_o))
                            # print('P_i=' + str(P_i))
                            # print('V_i=' + str(V_i))
                            # print('max_time='+str(max_time))
                            # print('on_the_ground='+str(on_the_ground))
                            a=np.dot(V_i, V_i)
                            b=2*np.dot(delta_P, V_i)
                            c=np.dot(delta_P, delta_P)-(k_factor * self.minimum_separation_distance)**2
                            det=b**2-4*a*c
                            if det>=0:
                                delay3 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                                return min(math.ceil(delay3 * 10) / 10, max_time)
                            else:
                                print('det<0, this should not happen')
                            return None
            else:
                # There is no delay that makes the closest point of approach be at that specific distance (i.e. if the agent was in the air there would be a conflict)
                # Since you're on the ground find when the intruder will be far enough that you can take off
                # TODO handle case when V_i is 0
                a = np.dot(V_i, V_i)
                b = 2 * np.dot(delta_P, V_i)
                c = np.dot(delta_P, delta_P) - (k_factor * self.minimum_separation_distance) ** 2
                det = b ** 2 - 4 * a * c
                if det >= 0:
                    delay3 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                    return min(math.ceil(delay3 * 10) / 10, max_time)
                else:
                    print('det<0, this should not happen')
                return None
        else:
            # in the air
            if debug:
                print('Delta is '+str(Delta))
                if delta_V_squared != 0:
                    print('a is '+str(a))
                    print('b is '+str(b))
                    print('c is '+str(c))
            if Delta >= 0 and a != 0:
                delay1 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                delay2 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if delay1 >= 0:
                    if debug:
                        print('looking at delay 1')
                    # The equation is only valid if the delay is smaller than the time of closest approach found on the next line
                    t_cpa_with_delay = -np.dot(delta_P + V_o * delay1, delta_V) / delta_V_squared
                    if delay1 <= t_cpa_with_delay:
                        if (0 <= t_cpa_stop < delay1 and d_cpa_stop < self.minimum_separation_distance) or d_initial < self.minimum_separation_distance:
                            # Sanity check
                            print('Cannot escape the conflict')
                            return None
                        else:
                            return min(math.ceil(delay1 * 10) / 10, max_time)
                    else:
                        print('delay 1 greater than the time of closest approach found, no solution')
                        return None
                else:
                    if delay2 < 0:
                        print('This should not happen if there is a conflict, delay2<0')
                    else:
                        if debug:
                            print('looking at delay 2')
                        # The equation is only valid if the delay is smaller than the time of closest approach found on the next line
                        t_cpa_with_delay = -np.dot(delta_P + V_o * delay2, delta_V) / delta_V_squared
                        if delay2 <= t_cpa_with_delay:
                            if (0 <= t_cpa_stop < delay2 and d_cpa_stop < self.minimum_separation_distance) or d_initial < self.minimum_separation_distance:
                                return None
                            else:
                                return min(math.ceil(delay2 * 10) / 10, max_time)
                        else:
                            # The solution delay2 is invalid
                            if closest_distance_stop <= self.minimum_separation_distance:
                                return None
                            else:
                                # Saved by the gong. Moving without delay will cause a conflict on the time segment.
                                # Waiting it out might not save you if the intruder keeps the same velocity, but if the intruder turns you have a chance
                                # Conflict is avoided on that time segment by waiting
                                return max_time
            else:
                # There is no delay that makes the closest point of approach be at that specific distance
                # This means that even by staying in place the two agents come close
                if closest_distance_stop < self.minimum_separation_distance:
                    return None
                else:
                    # Saved by the gong
                    return max_time

    def quadratic(self,start_point,end_point,cell_point):
        # Given a segment going from start_point to end_point, find the points between which the segment is closest
        # than the minimum separation distance to the cell point
        # returns the interval expressed in term of fraction (between 0 and 1)
        interval=[]
        length=np.linalg.norm(end_point-start_point)
        side = start_point - cell_point
        if length==0:
            # Agent is staying in place
            if np.linalg.norm(side) <= self.minimum_separation_distance:
                interval=[0,1]
        else:
            u =(end_point-start_point)/length
            b = np.dot(u,side)
            # Adding slightly superior to 1 k-factor for numerical issues (i.e. make the interval slightly bigger)
            c = np.dot(side,side)- (1.0001*self.minimum_separation_distance)**2
            delta = b**2-c
            if delta > 0:
                x1 = -b - math.sqrt(delta)
                x2 = -b + math.sqrt(delta)
                x1 = clamp(0,length,x1)
                x2 = clamp(0,length,x2)
                interval = [x1/length, x2/length]
        return interval

    def get_interval(self, position, time):
        i, j = self.get_coordinates(position)
        intervals = self.occupancy_grid[i, j]
        for interval in intervals:
            if interval[0] <= time <= interval[1]:
                return interval
        return None

    def get_coordinates(self,position):
        i=int(round(position[0]/self.cell_length))
        j=int(round(position[1]/self.cell_length))
        return i,j

    def get_position(self,i,j):
        x=i*self.cell_length
        y=j*self.cell_length
        return np.array([x,y])

    def is_path_available(self, ownship_start_time, start_point, end_point, end_time, debug=False):
        """ Assuming the ownship leaves the start_point at the start_time to finish at end_point and end_time, is there a conflict with other
        flight plans ?
        Returns available (Bool), intruder trajectory
        Returns only the section of the intruder path on which the collision will occur => the time interval returned is included in [ownship_start_time, end_time]"""
        # Naive approach
        # Iterate over all flight plans, find all flight segments that overlap in time with the input time
        # Compute for each of those segments the closest time of approach
        # Need to intersect time between time_start and time_end and get the resulting trajectory
        if debug:
            print('is path available time interval '+str([ownship_start_time,end_time]))
        ownship_velocity = (end_point-start_point)/(end_time - ownship_start_time)
        for flight_plan in self.flight_plans.values():
            trajectory, times = flight_plan.get_planned_trajectory_between(ownship_start_time, end_time)
            if times is not None and len(times)>=2 and abs(times[0]-times[1])<0.0001:
                print('there is an issue with get planned trajectory between')
                print(trajectory)
                print(times)
                flight_plan.get_planned_trajectory_between(ownship_start_time, end_time, debug=True)
            if trajectory is not None:
                n = len(trajectory)
                intruder_segment_start_pos = np.copy(trajectory[0])
                intruder_segment_start_time = times[0]
                analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity
                for i in range(1,n):
                    intruder_segment_end_pos = trajectory[i]
                    intruder_segment_end_time = times[i]
                    intruder_velocity = (intruder_segment_end_pos - intruder_segment_start_pos) / (intruder_segment_end_time - intruder_segment_start_time)
                    intruder_pos = intruder_segment_start_pos + (analysis_start_time - intruder_segment_start_time) * intruder_velocity
                    # Let's compute the closest time of approach
                    delta_v=intruder_velocity-ownship_velocity
                    delta_v_squared=np.dot(delta_v, delta_v)
                    delta_p=intruder_pos-ownship_pos
                    if delta_v_squared !=0:
                        t_cpa = analysis_start_time - np.dot(delta_v, delta_p) / delta_v_squared
                        # Let's find the time of closest approach on the segment
                        clamped_t_cpa=clamp(analysis_start_time,intruder_segment_end_time,t_cpa)-analysis_start_time # between 0 and the end of the segment (t_elapsed)
                    else:
                        # If ownship and intruder have the same velocity vector then the distance between them is constant
                        clamped_t_cpa = 0
                    closest = np.linalg.norm(intruder_pos+intruder_velocity*clamped_t_cpa-ownship_pos-ownship_velocity*clamped_t_cpa)
                    if closest < self.minimum_separation_distance:
                        if debug:
                            print('The closest point of approach is '+str(closest))
                            print('Total trajectory is '+str(trajectory))
                            print('Total times is ' + str(times))
                        if debug and abs(times[i-1]-times[i])<0.00001:
                            print('Problem with time')
                        if debug:
                            trajectory, times = flight_plan.get_planned_trajectory_between(ownship_start_time, end_time, debug=True)
                        return False, [trajectory[i-1:i+1], times[i-1:i+1]]
                    intruder_segment_start_time = intruder_segment_end_time
                    intruder_segment_start_pos = intruder_segment_end_pos
                    analysis_start_time = max(ownship_start_time, intruder_segment_start_time)
                    ownship_pos = start_point + (analysis_start_time - ownship_start_time) * ownship_velocity

        return True, None

    def add_flight_plan(self, flight_plan, id):
        self.flight_plans[id]=flight_plan
        old_pos=flight_plan.positions[0]
        old_time=flight_plan.times[0]
        for i in range(1, len(flight_plan.positions)):
            # The cells are in the grid coordinates, the flight plan is in world coordinates
            new_pos = flight_plan.positions[i]
            cells = self.on_the_grid_get_cells_along(old_pos, new_pos)
            new_time = flight_plan.times[i]
            dt = new_time - old_time
            for cell in cells:
                # Find the times where the cell is occupied by solving the quadratic equation
                cell_position = self.get_position(cell[0],cell[1])
                interval = self.quadratic(old_pos,new_pos,cell_position)
                if interval != []:
                    time_interval = [old_time+ dt * interval[0],
                                     old_time + dt * interval[1]]
                    self.update_interval(time_interval,cell[0],cell[1])
            old_pos = new_pos
            old_time = flight_plan.times[i]

    def terminate_flight_plan(self, id):
        del self.flight_plans[id]

    def on_the_grid_get_cells_along(self,position_a,position_b):
        """ Simplified algo for SIPP, always traveling between neighboring points, returns all cells when traveling diagonally """
        cells = []
        x_a = int(round(position_a[0]/self.cell_length))
        y_a = int(round(position_a[1]/self.cell_length))
        x_b = int(round(position_b[0]/self.cell_length))
        y_b = int(round(position_b[1]/self.cell_length))
        cells.append([x_a,y_a])
        cells.append([x_b, y_b])
        if x_a != x_b and y_a != y_b:
            cells.append([x_a, y_b])
            cells.append([x_b, y_a])
        return cells

    def get_cells_along(self,position_a,position_b):
        """ DDA (Digital Differential Analyzer) line algorithm. Implemented using an incremental error algorithm
        (with floats), to account for the fact that the trajectory does not snap on the grid
        Supercover version of the algorithm (returns all the grid cell that the line encounter and not just one per axis)
        Different sources used:
        http://eugen.dedu.free.fr/projects/bresenham/ (supercover version of Bresenham)
        http://groups.csail.mit.edu/graphics/classes/6.837/F01/Lecture04/lecture04.pdf (logic behind the algorithm)
        https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html (good step by step walkthrough of how the
        algorithm is modified to use integer arithmetic)
        Because of the floating aspect the code is not optimized (but it could be)
        Took way too long to implement because of dumb sign errors I made
        """
        cells=[]
        x_a=position_a[0]/self.cell_length
        y_a=position_a[1]/self.cell_length
        x_b = position_b[0]/self.cell_length
        y_b = position_b[1]/self.cell_length
        dx = x_b - x_a
        dy = y_b - y_a
        if dy < 0:
            y_step = -1
            dy = -dy
        else:
            y_step = 1
        if dx < 0:
            x_step = -1
            dx = -dx
        else:
            x_step = 1
        x = x_a
        y = y_a
        if dx >= dy:
            if dx != 0:
                m = (dy / dx )
                y=y_a+m*(round(x_a)-x_a) * x_step * y_step
                eps=y-round(y)
            else:
                # dx==0, since dx>=dy, dy=0
                cells.append([int(round(x)),int(round(y))])
                return cells
            x = round(x_a)
            y = y_a+m*(x-x_a)* x_step * y_step
            cells.append([int(round(x)),int(round(y))])
            error_prev = eps
            error = eps * y_step
            while -0.5 > x - round(x_b) or x-round(x_b) > 0.5:
                x += x_step
                if (error + m) > 0.5:
                    y += y_step
                    error += m - 1
                    if(error_prev + m/2) < 0.5:
                        cells.append([int(round(x)), int(round(y - y_step))])
                    elif (error_prev + m/2) > 0.5:
                        cells.append([int(round(x-x_step)), int(round(y))])
                else:
                    error += m
                cells.append([int(round(x)), int(round(y))])
                error_prev = error
        else:
            if dy != 0:
                m = dx / dy
                x = x_a + m * (round(y_a) - y_a) * x_step * y_step
                eps = x - round(x)
            y = round(y_a)
            x = x_a + m * (y - y_a) * x_step * y_step
            cells.append([int(round(x)), int(round(y))])
            error_prev = eps
            error = eps * x_step
            while -0.5 > y - round(y_b) or y-round(y_b) > 0.5:
                y += y_step
                if (error + m) > 0.5:
                    x += x_step
                    error += m - 1
                    if(error_prev + m/2) < 0.5:
                        cells.append([int(round(x - x_step)), int(round(y))])
                    elif (error_prev + m/2) > 0.5:
                        cells.append([int(round(x)), int(round(y-y_step))])
                else:
                    error += m
                cells.append([int(round(x)), int(round(y))])
                error_prev = error
        return cells

    def validate_flight_plan(self,flight_plan):
        return True


