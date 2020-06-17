# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:33:05 2020

@author: Kamil Chrustowski
"""
from math import sqrt
from mutation import *
from crossover import *
from numpy import array, count_nonzero, random
from selection import *
from utils import *
OPTIMAL_LENGTH = 869
EPSILON = 0.01
#genetic algorithm for travelling salesman problem

mapping = {
               0:(119, 38),
               1:(37, 38),
               2:(197, 55),
               3:(85, 165),
               4:(12, 50),
               5:(100, 53),
               6:(81, 142),
               7:(121, 137),
               8:(85, 145),
               9:(80, 197),
               10:(91, 176),
               11:(106, 55),
               12:(123, 57),
               13:(40, 81),
               14:(78, 125),
               15:(190, 46),
               16:(187, 40),
               17:(37, 107),
               18:(17, 11),
               19:(67, 56),
               20:(78, 133),
               21:(87, 23),
               22:(184, 197),
               23:(111, 12),
               24:(66, 178),
          }

def TSP_distance(genome: array):
    distances = []
    return sum([euclidean_distance(mapping[genome[i]], mapping[genome[i + 1]]) for i in range(len(genome) - 1)], 0)
               
def euclidean_distance(xy1, xy2):
    return sqrt((xy1[0] - xy2[0])**2 + ( xy1[1] - xy2[1] )**2)

def fitness(genome: array):
    total = TSP_distance(genome)
    return 1/total if len({num for num in genome}) == len(genome) and total != 0 else 0

def print_generation(generation:list, iteration:int):
    print("Generation for iteration No.: ", iteration)
    for genome in generation:
        print(genome,"|\t", "distance: ", TSP_distance(genome))  
    
def isEnd(generation: list):
    g = [fitness(genome) for genome in generation]
    logical = array(g) >= 1/869
    return count_nonzero(logical) >= len(generation)//2

#preparing generation
generation = [getShuffle(list(mapping.keys())) for i in range(100)]

#example for 100 iterations        
for i in range(10000):

    #elitism
    elitism = rank_selection(generation, fitness, 20)
    #select_temp_population
    generation = roulette_wheel_selection(generation, fitness, 80)
    #selection_crossover
    s = set()
    replacement = []
    for j in range(80):
        idx = random.randint(0, len(generation))
        while idx in s:
            idx = random.randint(0, len(generation))
        idx2 = random.randint(0, len(generation))
        while idx2 in s or idx2 == idx:
            idx2 = random.randint(0, len(generation))
        child = order_crossover(generation[idx], generation[idx2])
        replacement.append(child)
  
    generation = replacement
        
    #mutation
    for l in range(len(generation)):
        val = random.uniform()
        if val <= 0.01:
            random_swap(generation[l])

    #new population_with_old_ones
    for elit in elitism:
        generation.append(elit)
    if isEnd(generation):
        print_generation(generation, i+1)
        break