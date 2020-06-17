# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:59:50 2020

@author: Kamil Chrustowski
"""
from mutation import *
from crossover import *
from numpy import *
from selection import *
from utils import *
GOAL = 33
EPSILON = 0.01
#genetic algorithm for equation: 2a^2 + b = 33, where a,b in <0, 15>

def equation_value(genome: array):
    a = int("".join([ f'{num}' for num in genome[:4]]), 2)
    b = int("".join([ f'{num}' for num in genome[4:]]), 2)
    return abs(2 * (a**2) + b - GOAL)

def fitness(genome: array):
    val = equation_value(genome)
    return 1 / (1 + val)

def print_generation(generation:list, iteration:int):
    print("Generation for iteration No.: ", iteration)
    for genome in generation:
        print(genome,"|\t equation value: ", equation_value(genome))  
    
def isEnd(generation: list):
    g = [fitness(genome) for genome in generation]
    logical = abs(array(g) - 1.0) <= EPSILON
    return count_nonzero(logical) >= len(generation)//1.2

#preparing generation
generation = [random.randint(2, size=(1, 8))[0] for i in range(10)]

#example for 100 iterations        
for i in range(100):
    #select_population
    generation = roulette_wheel_selection(generation, fitness, 10)
    #selection_crossover
    s = set()
    replacement = []
    for j in range(5):
        idx = random.randint(0, len(generation))
        while idx in s:
            idx = random.randint(0, len(generation))
        idx2 = random.randint(0, len(generation))
        while idx2 in s or idx2 == idx:
            idx2 = random.randint(0, len(generation))
        child, _ = two_point_crossover(generation[idx], generation[idx2])
        replacement.append((idx, child))
    #replace
    for k, genome in replacement:
        generation[k] = genome
        
    #mutation
    for l in range(len(generation)):
        val = random.uniform()
        if val <= 0.1:
            random_replacement(generation[l])
    #new population
    
    print_generation(generation, i+1)
    if(isEnd(generation)):
        break