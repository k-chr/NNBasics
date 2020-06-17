# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:40:56 2020

@author: Kamil Chrustowski
"""
from numpy import *
from random import randint
"""
It does the chromosome inversion of existing genom
"""
def chromosome_inversion(genome:array)->array:
    if(len(genome) < 1): 
        return
    i = randint(0, len(genome) - 1)
    j = randint(0, len(genome) - 1)
    while i == j:
        j = randint(0, len(genome) - 1)
    i, j = (i, j) if i < j else (j, i)
    genome[i:j] = array(array(genome[i:j])[::-1])