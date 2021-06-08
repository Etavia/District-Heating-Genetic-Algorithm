# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:11:18 2021

@author: Tuomas Du-Ikonen
"""
import random
import numpy as np
import matplotlib.pyplot as plt

# Functions needed in algorithm
def create_starting_population(pop_size, num_genes, gene_val_low, gene_val_high):  
    population = np.zeros((3, pop_size, num_genes))
    
    for x in range(pop_size):
        for y in range(num_genes):
            population[0,x,y] = random.randint(gene_val_low, gene_val_high)
    return population

def set_plant_parameters(population):
    """
    Parameters of different power plants
    Index   Type                    Emissions   Cost
    1       Coal                    10          4
    2       Heavy Fuel Oil          8           5
    3       Natural Gas             5           5
    4       Wood pellet/chip        4           6
    5       Heat Pump               3           7
    6       Small Modular Nuclear   1           10
    7       Geothermal              1           8
    """
    
    pop_size = np.shape(population)[1]
    num_genes = np.shape(population)[2]
    
    for y in range(pop_size):
        for x in range(num_genes):
            if population[0,y,x] == 1:
                population[1,y,x] = 10
                population[2,y,x] = 4
            elif population[0,y,x] == 2:
                population[1,y,x] = 8
                population[2,y,x] = 5
            elif population[0,y,x] == 3:
                population[1,y,x] = 5
                population[2,y,x] = 5
            elif population[0,y,x] == 4:
                population[1,y,x] = 4
                population[2,y,x] = 6
            elif population[0,y,x] == 5:
                population[1,y,x] = 3
                population[2,y,x] = 7
            elif population[0,y,x] == 6:
                population[1,y,x] = 1
                population[2,y,x] = 10
            elif population[0,y,x] == 7:
                population[1,y,x] = 1
                population[2,y,x] = 8
    
    return population

def fitness_calculation(population):
    # Variables for sigmoid function
    varmin = 10.80542
    varmax = -0.069568
    n = 5.078503
    ec50 = 1806.296
    
    pop_size = np.shape(population)[1] # Get number of population
    num_genes = np.shape(population)[2] # Get number of genes (power plants)
    
    fitness = np.zeros((pop_size, 1))
    
    h = num_genes * 50 # Plant power combined in MW  
    
    for y in range(pop_size):
        plantemi = 0
        plantcost = 0
        for x in range(num_genes):
            plantemi = plantemi + population[1,y,x]
            plantcost = plantcost + population[2,y,x]
        heatscore = varmin + (varmax - varmin)/(1 + np.power(10, n * (np.log10(h) - np.log10(ec50)))) # Heat power fitness calculation
        emiscore = -1/60 * plantemi + 10 # Plant emission score
        costscore = -1/60 * plantcost + 10 # Plant cost score
        fitness[y,0] = heatscore + emiscore + costscore
    
    return fitness

def select_individual_by_tournament(population, fitness):
    # Get population size
    population_size = np.shape(population)[1]
    
    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)
    
    # Get fitness score for each
    fighter_1_fitness = fitness[fighter_1]
    fighter_2_fitness = fitness[fighter_2]
    
    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
    
    # Return the chromsome of the winner
    return population[0, winner, :]

def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)
    
    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
    child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    
    # Return children
    return child_1.reshape(1, chromosome_length), child_2.reshape(1, chromosome_length)

def randomly_mutate_population(population, mutation_rate):
    
    pop_size = np.shape(population)[1] # Get number of population
    num_genes = np.shape(population)[2] # Get number of genes (power plants)
    
    # Create new array with same dimensions than population array and then randomize new value for each cell.
    random_mutation_array = np.random.random(size=(pop_size, num_genes))
    # Create random array with true/false in each cell
    random_mutation_boolean = random_mutation_array <= mutation_rate
    
    # Go through each cell in array and if it's suppose to be mutated then randomize new value for it.
    for x in range(pop_size):
        for y in range(num_genes):
            if random_mutation_boolean[x,y] == 1: # If cell is true...
                population[0,x,y] = random.randint(gene_val_low, gene_val_high) # ...then create random variable to  it.
                if population[0,x,y] == 1: # Update mutated gene parametres
                    population[1,x,y] = 10
                    population[2,x,y] = 3
                elif population[0,x,y] == 2:
                    population[1,x,y] = 8
                    population[2,x,y] = 4
                elif population[0,x,y] == 3:
                    population[1,x,y] = 5
                    population[2,x,y] = 5
                elif population[0,x,y] == 4:
                    population[1,x,y] = 4
                    population[2,x,y] = 5
                elif population[0,x,y] == 5:
                    population[1,x,y] = 6
                    population[2,x,y] = 6
                elif population[0,x,y] == 6:
                    population[1,x,y] = 1
                    population[2,x,y] = 10
                elif population[0,x,y] == 7:
                    population[1,x,y] = 1
                    population[2,x,y] = 6
           
    return population
     

# Main algorithm

# Parametres
num_generations = 100 # Number of generations
pop_size = 30 # Population size
pop_low = 18 # Lowest reasonanble number of power plants
pop_high = 70 # Highest reasonable number of power plants
gene_val_low = 1 # Lowest gene value
gene_val_high = 7 # Highest gene value
mutation_rate = 0.002 # Mutation rate

# Reasonable number of 50 MW power plant units is from 18 to 60. For each of these numbers
# separate GA run needs to be made. Best result from each of the runs is stored to this array.
best_plant_qty_score_progress = []

best_generation = [] # Best generation is stored to this array

# Variables
best_ind_idx = 0 # Index of best individual
best_fitness = 0
lowest_fitness = 17.0

coal = 0
hfo = 0
ng = 0
bio = 0
hp = 0
smr = 0
gt = 0

fitarr = np.zeros((pop_high-pop_low,1))

for num_genes in range(pop_low, pop_high):
    
    # Create starting population
    population = create_starting_population(pop_size, num_genes, gene_val_low, gene_val_high)
    # Set plant parameters to population
    population = set_plant_parameters(population)
    
    # Calculate starting fitness
    fitness = fitness_calculation(population)
    
    # Add starting best score to progress tracker
    best_score_progress = [] # Tracks progress
      
    # Add best score to array
    best_score_progress.append(max(fitness_calculation(population)))
    
    for x in range(num_generations):
        # Create an empty list for new population
        new_population = np.zeros((3, num_genes))
        zero_row = np.zeros((2, num_genes))
        fitness = fitness_calculation(population)
        
        # Create new population generating two children at a time
        for i in range(int(pop_size/2)):
            parent_1 = select_individual_by_tournament(population, fitness)
            parent_2 = select_individual_by_tournament(population, fitness)       
            child_1, child_2 = breed_by_crossover(parent_1, parent_2)
            child_1 = np.vstack((child_1, zero_row))
            child_2 = np.vstack((child_2, zero_row))
            
            new_population = np.dstack((new_population, child_1))
            new_population = np.dstack((new_population, child_2))
        
        new_population = np.rot90(new_population, k=-1, axes=(1,2)) # Rotate array
        population = new_population
        population = np.delete(population, 0, 1) # Delete first row filled with zeros
        
        # Set plant parameters to population
        population = set_plant_parameters(population)
            
        # Apply mutation
        population = randomly_mutate_population(population, mutation_rate)
        
        # Recalculate fitness after mutation
        fitness = fitness_calculation(population) 
        
        # Record best score
        best_score_progress.append(max(fitness))
            
    # Add best score to quantity of power plants tracker.    
    best_plant_qty_score_progress.append(max(fitness_calculation(population)))
    
    fitarr[num_genes-pop_low,0] = max(fitness)
    
    if max(fitness) >= max(best_plant_qty_score_progress):
        best_of_best_score_progress = best_score_progress
        best_fitness = max(fitness)
        best_ind_idx = np.argmax(fitness) # Get index of best fitness
        best_generation = population[0,best_ind_idx,:] # Add best generation to array
    
    maxfit = max(fitness)
    maxplant = max(best_plant_qty_score_progress)
        
    print("Computing. Please wait...")
    
    """
    # Best generation parsed to check that code and functions are working
    bestgen = population[0:3,best_ind_idx:best_ind_idx+1,0:num_genes]
    bestfit = fitness_calculation(bestgen)
    """
    
print("")
print("Highest fitness in end:", max(best_plant_qty_score_progress))

plantlist = np.array([[1, 0],
                      [2, 0],
                      [3, 0],
                      [4, 0],
                      [5, 0],
                      [6, 0],
                      [7, 0]])        

for x in range(len(best_generation)):
    if best_generation[x] == 1:
        coal += 1
    if best_generation[x] == 2:
        hfo += 1
    if best_generation[x] == 3:
        ng += 1
    if best_generation[x] == 4:
        bio += 1
    if best_generation[x] == 5:
        hp += 1
    if best_generation[x] == 6:
        smr += 1
    if best_generation[x] == 7:
        gt += 1
     
plantlist[0,1] = coal
plantlist[1,1] = hfo
plantlist[2,1] = ng
plantlist[3,1] = bio
plantlist[4,1] = hp
plantlist[5,1] = smr
plantlist[6,1] = gt

ordered = plantlist[np.argsort(plantlist[:, 1])[::-1]]

for x in range(len(ordered)):
    ordered[x,1] = ordered[x,1] * 50

ord2 = np.array([["Wood pellet/chippp", ordered[0,1]],
                 ["Wood pellet/chippp", ordered[1,1]],
                 ["Wood pellet/chippp", ordered[2,1]],
                 ["Wood pellet/chippp", ordered[3,1]],
                 ["Wood pellet/chippp", ordered[4,1]],
                 ["Wood pellet/chippp", ordered[5,1]],
                 ["Wood pellet/chippp", ordered[6,1]]])

for x in range(len(ordered)):
    if ordered[x,0] == 1:
        ord2[x,0] = "Coal\t\t\t\t"
    if ordered[x,0] == 2:
        ord2[x,0] = "HFO\t\t\t\t\t"
    if ordered[x,0] == 3:
        ord2[x,0] = "Natural gas\t\t\t"
    if ordered[x,0] == 4:
        ord2[x,0] = "Wood pellet/chip\t"
    if ordered[x,0] == 5:
        ord2[x,0] = "Heat pump\t\t\t"
    if ordered[x,0] == 6:
        ord2[x,0] = "SMR\t\t\t\t\t"
    if ordered[x,0] == 7:
        ord2[x,0] = "Geothermal\t\t\t"


print("")
print("Best combination of plants")
print(ord2[0,0], ord2[0,1], "MW")
print(ord2[1,0], ord2[1,1], "MW")
print(ord2[2,0], ord2[2,1], "MW")
print(ord2[3,0], ord2[3,1], "MW")
print(ord2[4,0], ord2[4,1], "MW")
print(ord2[5,0], ord2[5,1], "MW")
print(ord2[6,0], ord2[6,1], "MW")

for x in range(pop_low):
    best_plant_qty_score_progress.insert(0,0.0)

plt.plot(best_plant_qty_score_progress)
plt.xlabel('Number of power plant units')
plt.ylabel('Best score')
plt.xlim([pop_low,pop_high])
plt.ylim([lowest_fitness,best_fitness+1])
plt.show()

plt.plot(best_of_best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best score')
plt.show()
