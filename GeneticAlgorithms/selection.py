# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:20:06 2020

@author: Kamil Chrustowski
"""
from random import randint, uniform, random
from utils import *
from numpy import *

def roulette_wheel_selection(population:list, fitness_function, n):
    fit_values = [(genome, fitness_function(genome)) for genome in population]
    fit_sum = sum([values[1] for values in fit_values])
    indices = set()
    subpopulation = []
    while len(indices) != n:
        val = uniform(0, fit_sum)
        curr = 0
        for index, (genome, value) in enumerate(fit_values):
            curr += value
            if curr > val and index not in indices:
                indices.add(index)
                subpopulation.append(genome)
                break
    return subpopulation

def rank_selection(population: list, fitness_function, n):
    fit_values = [(genome, fitness_function(genome)) for genome in population]
    q = PriorityQueue()
    for genom, value in fit_values:
        q.push(genom, value) 
    return [q.pop() for i in range(n)]

def tournament_selection(population: list, fitness_function, n):
    fit_values = [(genome, fitness_function(genome)) for genome in population]  
    assert n <= len(population) - 1
    tournaments = []
    for i in range(n):
        q = PriorityQueue()
        tournaments.append(q)
    i = 0
    while len(fit_values) > 0:
        if i%n == 0 and i != 0:
            i = 0
        val = randint(0, len(fit_values) - 1)
        genome, value = fit_values.pop(val)
        tournaments[i].push(genome, value)
        i += 1
        
    return [tournament.pop() for tournament in tournaments]