# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:41:34 2020

@author: Kamil Chrustowski
"""

from mutation import *
from crossover import *
from numpy import array, count_nonzero, random
from selection import *
from utils import *
MAX_WEIGHT = 35
EPSILON = 0.01
#genetic algorithm for knapsack problem

mapping = {
               0:(3, 266),
               1:(13, 442),
               2:(10, 671),
               3:(9, 526),
               4:(7, 388),
               5:(1, 245),
               6:(8, 210),
               7:(8, 145),
               8:(2, 126),
               9:(9, 322)
          }

def knapsack_weight(genome: array):
    return sum([mapping[i][0] for i, v in enumerate(genome) if v == 1], 0)

def knapsack_value(genome: array):
    return sum([mapping[i][1] for i, v in enumerate(genome) if v == 1], 0)

def fitness(genome: array):
    return knapsack_value(genome) if knapsack_weight(genome) <= MAX_WEIGHT else 0

def print_generation(generation:list, iteration:int):
    print("Generation for iteration No.: ", iteration)
    for genome in generation:
        print(genome,"|\t knapsack value: ", knapsack_value(genome), " , knapsack weight: ", knapsack_weight(genome), ", fitness: ", fitness(genome))  
    
def isEnd(generation: list):
    g = [fitness(genome) for genome in generation]
    logical = array(g) >= 2222
    return count_nonzero(logical) >= len(generation)

#preparing generation
generation = [random.randint(2, size=(1, 10))[0] for i in range(8)]

#example for 100 iterations        
for i in range(10000):

    #elitism
    elit1, elit2 = rank_selection(generation, fitness, 2)
    #select_temp_population
    generation = roulette_wheel_selection(generation, fitness, 6)
    #selection_crossover
    s = set()
    replacement = []
    for j in range(6):
        idx = random.randint(0, len(generation))
        while idx in s:
            idx = random.randint(0, len(generation))
        idx2 = random.randint(0, len(generation))
        while idx2 in s or idx2 == idx:
            idx2 = random.randint(0, len(generation))
        child, _ = two_point_crossover(generation[idx], generation[idx2])
        replacement.append(child)
  
    generation = replacement
        
    #mutation
    for l in range(len(generation)):
        val = random.uniform()
        if val <= 0.05:
            random_replacement(generation[l])
            
    generation.append(elit1)
    generation.append(elit2)
    #new population
    
    print_generation(generation, i+1)
    if isEnd(generation):
        break