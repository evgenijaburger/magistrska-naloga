from simulation_code import *

import time
import datetime

"""
How to run a simulation.
"""

DtSimulations = 8.5*3

# since it would be too much to run all the simulations at once we limit the space of our parameters
contractility = [1]
resistance = [1]
compliance = [1]


triples = []

for C in contractility:
    for R in resistance:
        for S in compliance:
            triples.append([C, R, S])

# initialize json file { "C = 0.1" : []}
with open(f"simulations/simulation_results/example_results/results_{C}_{R}_{S}.json", 'w') as file:
    json.dump([], file, indent=4)

with open(f"simulations/simulation_results/example_results/results_{C}_{R}_{S}.json", 'w') as file:
    json.dump([], file, indent=4)

# run the simulations
run_multiple_simulations(triples, DtSimulations, f"simulations/simulation_results/example_results/results_{C}_{R}_{S}", f"simulations/simulation_results/example_results/results_{C}_{R}_{S}.json")
