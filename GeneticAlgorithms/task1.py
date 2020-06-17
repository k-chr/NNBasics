# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:05:43 2020

@author: Kamil Chrustowski
"""
from mutation import *
from crossover import *
from numpy import *
from selection import *
from utils import *

def print_generation(generation:list, iteration:int):
    print("Generation for iteration No.: ", iteration)
    for genome in generation:
        print(genome,"|\t", count_bits(genome)/len(genome))  
        
def isEnd(generation: list):
    g = [count_bits(genome)/len(genome) for genome in generation]
    logical = array(g) >= 1.0
    return count_nonzero(logical) >= len(generation)//2
    
#preparing generation
generation = [random.randint(2, size=(1, 10))[0] for i in range(10)]


for i in range(100):
    
    #selection
    father, mother = rank_selection(generation, count_bits, 2)
    #crossover
    child1, child2 = two_point_crossover(father, mother)
    #mutation
    val = random.uniform()
    if val <= 0.6:
        random_replacement(child1)
    val = random.uniform()
    if val <= 0.6:
        random_replacement(child2)
    #new population
    generation = rank_selection(generation, count_bits, 8)
    generation.append(child1)
    generation.append(child2)
    print_generation(generation, i+1)
    if(isEnd(generation)):
        break
    