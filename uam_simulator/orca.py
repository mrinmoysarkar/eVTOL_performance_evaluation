import numpy as np
import math

""" Implementation of the ORCA algorithm 
    Resources: van den Berg, Reciprocal n-body Collision Avoidance,
                RVO2 library (C++) https://github.com/snape/RVO2/blob/master/src/Agent.cpp
                Pyorca library to see another python implementation https://github.com/Muon/pyorca
    Other resources included in the code """


class Line:
    def __init__(self, point=np.array([0, 0]), direction=np.array([0, 0])):
        self.point = point
        self.direction = direction


class ORCA:
    def __init__(self):
        self.max_distance_to_neighbors = 5000  # m
        self.time_horizon = 250  # in seconds, the range to query/sense neighbors is 5 km, the max speed is 20 m/s => time_horizon = range/max_speed (250 seconds ~ 4.2 minutes)
        self.inv_time_horizon = 1 / self.time_horizon
        self.epsilon = 0.00001
        self.k_neighbors = 10

    def compute_new_velocity(self, agent, dt):
        # Compute the new velocity for the agent
        neighbors = agent.get_nearest_neighbors(self.k_neighbors, self.max_distance_to_neighbors)
        orca_lines =[]
        R = agent.radius * 1.1  # Slight increase to take care of floating point errors
        # Compute the agent desired velocity
        direction_to_goal = agent.goal - agent.position
        distance_to_goal = np.linalg.norm(direction_to_goal)
        pref_vel = min(agent.maxSpeed, distance_to_goal / dt) * direction_to_goal / distance_to_goal
        for neighbor in neighbors:
            # Construct ORCA lines
            # Project on the half plane
            rel_position = neighbor.position-agent.position
            d = np.linalg.norm(rel_position)
            # Opposite for some reason
            # Relative velocity computed using current velocity
            rel_velocity = agent.velocity - neighbor.velocity
            # Is there a collision
            if d > R:
                # No collision right now
                # Vector from cutoff center to relative velocity
                w = rel_velocity - self.inv_time_horizon * rel_position
                dot_product1 = np.dot(w, rel_position)
                w_norm = np.linalg.norm(w)
                unit_w = w / w_norm
                # dot_product1<0 is a necessary but not sufficient condition for the projection having to be done on the cutoff circle
                # The second condition compares the angle between w and - rel_postion (lambda) to the angle between rel_position and the radius tangent to the line (alpha)
                # Project on the circle if lambda < alpha => cos^2 (lambda) > cos^2(alpha) because alpha and lambda are between 0 and pi/2
                # cos^2 lambda = dot_product1**2/(|w|^2*|rel_position|^2)
                # cos^2 alpha = R^2 / |rel_position|^2
                if dot_product1 < 0 and dot_product1**2 > R**2 * np.dot(w, w):
                    # Should project on cut-off circle
                    # U is in the direction of w and the remaining distance to exit the cutoff circle
                    u = (R * self.inv_time_horizon - w_norm) * unit_w
                    direction = np.array([unit_w[1], -unit_w[0]])
                else:
                    # Need to project on legs
                    leg = math.sqrt(d**2 - R**2)
                    if np.linalg.det([rel_position,w]) > 0:
                        # On left leg, find direction by multiplying by rotation matrix (cone haf angle theta such that sin theta = R / d and cos theta = leg / d
                        # Unit vector
                        direction = np.array([rel_position[0]*leg - rel_position[1] * R,
                                             rel_position[0]*R + rel_position[1] * leg ]) / d**2
                    else:
                        # On right leg
                        direction = - np.array([rel_position[0] * leg + rel_position[1] * R,
                                              - rel_position[0] * R + rel_position[1] * leg]) / d**2
                    # Project the relative velocity on the leg
                    dot_product2 = rel_velocity * direction
                    # u is the point from the rel_velocity to the boundary
                    u = dot_product2 * direction - rel_velocity
            else:
                # Already colliding with neighbor, pick the velocity that will get us out within the time step
                w = rel_velocity - rel_position / dt
                w_norm = np.linalg.norm(w)
                unit_w = w / w_norm
                u = (R / dt - w_norm) * unit_w
                direction = np.array([unit_w[1], -unit_w[0]])
            line = Line(agent.velocity + u / 2, direction)
            orca_lines.append(line)
        # Try to solve the linear program, optimize for the preferred velocity of the agent
        line_fail, new_vel = self.linear_program2(orca_lines, agent.maxSpeed, pref_vel, False)

        if line_fail < len(orca_lines):
            # The feasible region is empty
            new_vel = self.linear_program3(orca_lines, line_fail, agent.maxSpeed, pref_vel)

        return new_vel

    # de Berg, Cheong, van Kreveld and Overmars, Computational Geometry: Algorithms and Applications, Third edition
    # Chapter 4: linear programming
    # 4.3 Incremental Linear Programming

    def linear_program1(self, lines, line_no, max_speed, opt_v, opt_dir):
        # Called when current velocity violates constraint line_no and we are looking for a solution on the constraint line_no
        # This is a 1D linear problem
        # returns False and the original velocity if the program is unfeasible
        # returns True and the new velocity which lays on constraint line_no

        # Initialize the bounds of the solution (scalar value indicating how far from point along direction is the solution)
        # This solver does not work for unbounded solutions but the solution is bounded by the max speed anyway
        # We are looking for the intersection of the line with the circle: |P + t* d|= R, this results in a quadratic equation
        # Where P is the point and d the direction of the constraint, |d|=1
        dot_product = np.dot(lines[line_no].point, lines[line_no].direction)
        discriminant = dot_product ** 2 - np.dot(lines[line_no].point, lines[line_no].point) + max_speed**2
        if discriminant < 0:
            # The constraint is outside the max speed limit
            return False, opt_v
        sqrt_discriminant = math.sqrt(discriminant)
        t_left = - dot_product - sqrt_discriminant
        t_right = - dot_product + sqrt_discriminant

        for i in range(0, line_no):
            # Find intersection of both lines https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
            denom = np.linalg.det([lines[line_no].direction, lines[i].direction])
            numer = np.linalg.det([lines[i].direction,lines[line_no].point - lines[i].point])
            if abs(denom)<= self.epsilon:
                # lines are almost parallel
                if numer < 0:
                    # if line_no is in the forbidden area of i and hence there are no solution
                    # I think it should not happens given how the solution is computed
                    return False, opt_v
                else:
                    continue
            t = numer / denom
            if denom >= 0:
                # Line i bounds line_no on the right
                t_right = min(t_right, t)
            else:
                # Line i bounds line_no on the right
                t_left = max(t_left, t)

            if t_left > t_right:
                # There is no solution on the constraint
                return False, opt_v

        if opt_dir:
            # We are just looking for a feasible solution
            if np.dot(opt_v, lines[line_no].direction) > 0:
                # Take right extreme
                new_vel = lines[line_no].point + t_right * lines[line_no].direction
            else:
                new_vel = lines[line_no].point + t_left * lines[line_no].direction
        else:
            # Should we use t_left or t_right as the new solution?
            # Project the optimal velocity on the constraint
            t = np.dot (lines[line_no].direction, opt_v - lines[line_no].point)
            if t < t_left:
                # projected to the left of the limit => the left limit is the new optimal
                new_vel = lines[line_no].point + t_left * lines[line_no].direction
            elif t > t_right:
                # projected to the right of the limit => the left limit is the new optimal
                new_vel = lines[line_no].point + t_right * lines[line_no].direction
            else:
                # the projection is in the middle of the limits
                new_vel = lines[line_no].point + t * lines[line_no].direction

        return True, new_vel

    def linear_program2(self, lines, max_speed, opt_v, opt_dir):
        """Solves the 2D linear program defined by the lines
        If there is a solution returns the number of constraint
        If there is no solution returns the constraint number that made the solution space empty
        If opt_dir is True then opt_v must be unit length
        """
        # Start by setting the desired solution
        # Not quite sure what opt_dir does, opt_dir is set to True when the solution space is empty
        if opt_dir:
            # opt_v is unit length
            result = max_speed * opt_v
        elif np.linalg.norm(opt_v) > max_speed:
            result = max_speed * opt_v / np.linalg.norm(opt_v)
        else:
            result = opt_v

        # Solve the linear program
        for i in range(0, len(lines)):
            line = lines[i]
            # Does the current result satisfy the constraint?
            if np.linalg.det([line.direction, line.point - result]) > 0:
                # The current result does not satisfy the new constraint, hence the new result will be on the new constraint unless the program is unfeasible
                feasible, new_result = self.linear_program1(lines, i, max_speed, opt_v, opt_dir)
                if not feasible:
                    # Constraint i makes the solution space empty, the result velocity will be ignored
                    return i, opt_v
                else:
                    result = new_result
        return len(lines), result

    def linear_program3(self, lines, line_fail, max_speed, opt_v, num_obstacles=0):
        # Called if the original 2D linear program is unfeasible see section 5.3 of ORCA paper
        # Basically the constraints are moved away until one velocity becomes feasible (it's a 3D linear programming problem)
        # The 2D linear program to solve is how much to relax
        # For now there are no non-participating or static obstacles (num obstacles  is 0)
        # Distance is the distance that constraints have to be relaxed by to open the solution space
        distance = 0
        result = opt_v
        # Start at the line that failed
        for i in range(line_fail,len(lines)):
            if np.linalg.det([lines[i].direction, lines[i].point - result])> distance:
                # the result does not satisfy line i constraint (even accounting for relaxation)
                # Since result is the only point in the relaxed solution space this means the solution space is empty

                # Treat constraints coming from non-participating traffic differently
                # Copy all lines that are not going to be relaxed
                if num_obstacles !=0:
                    # TODO Copy lines created by non-participating obstacles
                    projected_lines = []
                    pass
                else:
                    projected_lines=[]
                # Go through all constraints generated by participating traffic that are before the current constraint
                for j in range(num_obstacles,i):
                    new_line = Line()
                    # Find intersection of both lines https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                    denom = np.linalg.det([lines[i].direction, lines[j].direction])
                    if abs(denom) <= self.epsilon:
                        # lines are basically parallel
                        if np.dot(lines[i].direction,lines[j].direction)>0:
                            # lines are in the same direction
                            continue
                        else:
                            # lines are in opposite directions
                            # Set a point in the middle to find how much you have to relax the constraint to open the solution space
                            new_line.point = 0.5*(lines[i].point+lines[j].point)
                    else:
                        # Find the intersection of those two lines
                        numer = np.linalg.det([lines[j].direction, lines[i].point - lines[j].point])
                        new_line.point = lines[i].point + (numer / denom) * lines[i].direction
                    new_line.direction = (lines[j].direction-lines[i].direction)/ np.linalg.norm(lines[j].direction-lines[i].direction)
                    projected_lines.append(new_line)

                # We know that all the previous constraints left a non-empty solution space (since line_fail is the first constraint that emptied the solution space)
                # The new lines' direction is opposite to the directions along which the intersection of line_fail and constraint i move when the constraints are being relaxed
                projected_vel = np.array([-lines[i].direction[1],
                                          lines[i].direction[0]])
                # Solve linear program to figure out how much to relax the constraints
                # Projected_vel gives the direction along which to optimize
                line_fail, solution = self.linear_program2(projected_lines, max_speed, projected_vel, True)

                if line_fail < len(projected_lines):
                    # Failed to relax (this should not happen) if it fails it's because of floating point errors
                    pass
                result = solution
                # Set the relaxation distance based on the result of the linear program (the following formula is the signed distance from result to line i because direction is a unit vector)
                distance = np.linalg.det([lines[i].direction, lines[i].point - result])

        return result


