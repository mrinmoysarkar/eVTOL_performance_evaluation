from queue import Queue
from uam_simulator import simulation
from uam_simulator import display
import random
import json

# Simulation parameters
random.seed(77)  # Set a random seed to ensure repeatability of a given run
simulation_name = 'example_run'  # A string, will be used to name the log file

minimum_separation = 500  # m, minimum distance that agents must maintain between each other
length_arena = 139000  # m, size of the square simulation area
max_speed = 45  # m/s, maximum velocity of the agents
sensing_radius = 5000

time_step = 1 # Set to 10 for strategic agents, 1 for reactive agents
simulation_length = float('inf')  # Set to infinity to simulate until n_agents_results agents that were created after t0 have exited the simulation
n_agent_results = 100  # See above

n_intruder = 15  # Number of agents in the simulation (simulation starts at 0 agents and adds agent at a constant rate until n_intruder is achieved)
simulation_type = 'strategic'  # set to 'reactive' (access control: free) or 'strategic' (access control: 4DT contract)
algo_type = 'Decoupled'  # For reactive: MVP_Bluesky, ORCA, straight. For strategic: Decoupled, LocalVO, SIPP
structure = None  # None or {'type': 'layer', 'parameters': [0, 180]}. The range for the layer is specified in degrees in parameters and can be set to any number

visualization = True  # Turn visualization on or off. If visualization is on, the file must be saved in the simulation thread

if visualization:
    update_queue = Queue()
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
                                update_queue=update_queue,
                                stdout_to_file=True)
    dis = display.Display(update_queue, length_arena, display_update=20 * time_step)
    sim.run()
    dis.run()
else:
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
                                save_file=False,
                                n_valid_agents_for_simulation_end=n_agent_results,
                                stdout_to_file=True)
    log = sim.run()
    filename = 'logs/' + simulation_name + '.json'
    with open(filename, 'w') as file:
        json.dump(log, file, indent=4)
