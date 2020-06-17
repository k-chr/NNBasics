# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:40:56 2020

@author: Kamil Chrustowski
"""
import heapq
from numpy import *
from random import randint, uniform, random
"""
It does the chromosome inversion of existing genom
"""
def chromosome_inversion(genome:array):
    if(len(genome) < 1): 
        return
    i = randint(0, len(genome) - 1)
    j = randint(0, len(genome) - 1)
    while i == j:
        j = randint(0, len(genome) - 1)
    i, j = (i, j) if i < j else (j, i)
    genome[i:j] = array(array(genome[i:j])[::-1])
    
def adjacent_swap(genome: array):
    i = randint(0, len(genome) - 2)
    tmp = genome[i]
    genome[i] = genome[i+1]
    genome[i+1] = tmp
    
def random_swap(genome:array):
    i = randint(0, len(genome) - 1)
    j = randint(0, len(genome) - 1)
    while i == j:
        j = randint(0, len(genome) - 1)
    tmp = genome[i]
    genome[i] = genome[j]
    genome[j] = tmp
    
def random_replacement(genome: array):
    i = randint(0, len(genome) - 1)
    genome[i] = 1 - genome[i]
    
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
        
    
def count_bits(genome: array):
    val = count_nonzero(genome)
    return val
    
class PriorityQueue:

    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        heapq._heapify_max(self.heap)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq._heappop_max(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):

        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq._heapify_max(self.heap)
                break
        else:
            self.push(item, priority)