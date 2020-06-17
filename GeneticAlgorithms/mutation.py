# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:18:57 2020

@author: Kamil Chrustowski
"""
from numpy import *
from random import randint

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