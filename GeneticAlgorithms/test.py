# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:14:39 2020

@author: Kamil Chrustowski
"""

from mutation import *
from selection import *
from utils import *
from numpy import array

arr = array([0,0,0,1,1,0,0,1,0,1])
arr1 = array([0,1,0,1,1,0,0,1,0,1])
arr2 = array([0,0,0,1,1,0,0,0,0,1])
arr3 = array([0,0,0,0,0,0,0,1,0,1])
arr4 = array([1,1,0,1,1,0,0,1,0,1])
print(arr)
chromosome_inversion(arr)
print(arr)
l = [arr,arr1, arr2, arr3, arr4]
for genome in tournament_selection(l, count_bits, 2):
    print(genome)