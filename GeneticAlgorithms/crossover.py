# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:23:07 2020

@author: Kamil Chrustowski
"""
from numpy import *
from random import randint

def one_point_crossover(father:array, mother:array)->tuple:
    child1 = array(father)
    child2 = array(mother)
    val = randint(0, len(child1) - 1)
    child1[val:], child2[:val] = child2[val:], child1[val:]

def two_point_crossover(father: array, mother: array)->tuple:
    child1 = array(father)
    child2 = array(mother)
    start_ind = randint(0, len(child1) - 1)
    end_ind = randint(0, len(child1) - 1)
    while start_ind == end_ind:
        end_ind = randint(0, len(child1) - 1)
        
    child1[start_ind:end_ind], child2[start_ind:end_ind] = child2[start_ind:end_ind], child1[start_ind:end_ind]
    return (child1, child2)

def order_crossover(father: array, mother: array)->object:
    child = array(father)
    start_ind = randint(0, len(child) - 1)
    end_ind = randint(0, len(child) - 1)
    
    while start_ind == end_ind:
        end_ind = randint(0, len(child) - 1)
    print(f"start: {start_ind}\nend: {end_ind}")
    if start_ind > end_ind:
        start_ind, end_ind = end_ind, start_ind
        
    child[:start_ind] = -1
    child[end_ind:] = -1
    print("child: ", child)
    print("mother: ", mother)
    for i in range(start_ind):
        j = 0
        while mother[j] in child:
            j+=1
        child[i] = mother[j]
    for i in range(end_ind, len(child)):
        j = 0
        while mother[j] in child:
            j+=1
        child[i] = mother[j] 
    return child
    
    