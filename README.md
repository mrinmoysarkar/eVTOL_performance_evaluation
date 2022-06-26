# A Data-Driven Approach for Performance Evaluation of Autonomous eVTOLs

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
  - Tensorflow 2.8

N.B.: The code is tested on Python 3.7

# Experiments
1. To run a simple, simulation run the simple_UTM_sim.py. It will show the simulation GUI. 
You can change the configuration of the simulation by changing the parameters inside simple_UTM_sim.py.

2. To generate data from this UTM simulator you can run sim_data_generation.py

3. To generate eVTOL performance data run test_vehicle_peformance_lift_and_cruise.py, or test_vehicle_peformance_multicopter.py, or test_vehicle_peformance_vector_thrust.py  

N.B.:The UTM simulator generates simulation data in JSON format that needs to be converted to CSV file before it can be used by the eVTOL performance evaluator script (test_vehicle_performance_*.py). So, use convert_json_to_csv.py to convert all the JSON log files and then run test_vehicle_performance_*.py  

# UAM Database link
The generated dataset from this simulation experiments can be found in the following link:

1. UTM dataset: https://drive.google.com/drive/folders/1ty6J5D5wq1vYP6bS8z8jRp9wJTfxeykZ?usp=sharing
   
   File Name: all_UTM_sim_data.csv

2. All Trajectory dataset: https://drive.google.com/drive/folders/1ty6J5D5wq1vYP6bS8z8jRp9wJTfxeykZ?usp=sharing 
    
   File Name: all_trajectories.zip

3. eVTOL performance dataset: https://drive.google.com/drive/folders/1ty6J5D5wq1vYP6bS8z8jRp9wJTfxeykZ?usp=sharing
   
   Folder Name: profiles_eval

Researchers can start their experiments and analysis from the dataset instead of running the simulation to generate the dataset.
But researchers are also encouraged to implement new features on the simulator to make the simulation more realistic for the development of novel concepts in UAM domain. 

# visualization of the integrated simulation software (UTM+eVTOL performance analysis) 


https://user-images.githubusercontent.com/12240371/175831157-6e86c89e-02ba-4bab-bf48-85e1e269a2de.mp4



# CASE STUDY on the dataset
Two case study on the UTM dataset is shown in data_analysis_of_UTM_sim_data.ipynb 

N.B. To run the data_analysis_of_UTM_sim_data.ipynb Please download all_UTM_sim_data.csv and change the data_df = pd.read_csv('./logs/all_UTM_sim_data.csv') with the proper directory in the notebook.

# Clustering analysis
The clustering analysis code is in ./Clustering folder and corresponding code is in clustering.py file.

# Machine Learning based approach
The implementation of the machine learning based approach (training and testing) can be found in ML_based_eVTOL_pe.ipynb

# Other comments
1. The UTM simulator is developed by utilizing libraries developed in the following project:
https://github.com/colineRamee/UAM_simulator_scitech2021

2. The eVTOL performance is calculated using SUAVE libraries (https://github.com/suavecode/SUAVE).
A version of the SUAVE(2.5) is included in this project. So no installation is required for the SUAVE libraries.

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
