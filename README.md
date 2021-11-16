# ePE: a simulation framework for generation of benchmark database to evaluate eVTOL performance in executing UAM mission

It is a stand alone software package fully developed in Python for research purpose. To run the code follow the bellow instructions:

# Python Dependencies:
Install the following Python packages using pip: 
  - gurobi
  - numpy
  - pandas
  - jupyter notebook
  - scikit-learn
  - scipy
  - zlib
N.B.: The code is tested on Python 3.7

# Experiments
1. To run a simple simulation run the simple_UTM_sim.py. It will show the simulation GUI. 
You can change the configuration of the simulation by changing the parameters inside simple_UTM_sim.py.

2. To generate data from this UTM simulator you can run sim_data_generation.py

3. To generate eVTOL performance data run test_vehicle_performance.py 

N.B.:The UTM simulator generates simulation data in JSON format that needs to be converted to CSV file before it can be used by the eVTOL performance evaluator script (test_vehicle_performance.py). So, use convert_json_to_csv.py to convert all the JSON log files and then run test_vehicle_performance.py  

# UAM Database link
The generated dataset from this simulation experiments can be found in the following link:

Researchers can start their experiments and analysis from the dataset instead of running the simulation to generate the dataset.
But researchers are also encouraged to implement new features on the simulator to make the simulation more realistic for development of novel concepts in UAM domain.  

# CASE STUDY on the dataset
Two case study on the UTM dataset is shown in data_analysis_of_UTM_sim_data.ipynb 


# Other comments
1. The UTM simulator is developed by utilizing libraries developed in the following project:
https://github.com/colineRamee/UAM_simulator_scitech2021

2. The eVTOL performance is calculated using SUAVE libraries (https://github.com/suavecode/SUAVE).
A version of the SUAVE(2.4) is included in this project. So no installation is required for the SUAVE libraries.

# References
Please cite the following papers if you are using this code or the dataset:

Bibtex formatted citation:

1. @misc{sarkar2021framework, 
   title={A Framework for eVTOL Performance Evaluation in Urban Air Mobility Realm}, 
   author={Mrinmoy Sarkar and Xuyang Yan and Abenezer Girma and Abdollah Homaifar}, 
   year={2021}, 
   eprint={2111.05413}, 
   archivePrefix={arXiv}, 
   primaryClass={cs.RO} 
   }

2. @inproceedings{ramee2021development,
   title={Development of a Framework to Compare Low-Altitude Unmanned Air Traffic Management Systems},
   author={Ramee, Coline and Mavris, Dimitri N},
   booktitle={AIAA Scitech 2021 Forum},
   pages={0812},
   year={2021}
   }

3. @inbook{SUAVE2017,
   author = {Timothy MacDonald and Matthew Clarke and Emilio M. Botero and Julius M. Vegh and Juan J. Alonso},
   title = {SUAVE: An Open-Source Environment Enabling Multi-Fidelity Vehicle Optimization},
   booktitle = {18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference},
   chapter = {},
   pages = {},
   doi = {10.2514/6.2017-4437},
   URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2017-4437},
   eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2017-4437}
   }