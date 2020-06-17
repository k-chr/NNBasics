# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:14:39 2020

@author: Kamil Chrustowski
"""


from utils import chromosome_inversion
from numpy import array

arr = array([0,0,0,1,1,0,0,1,0,1])
print(arr)
chromosome_inversion(arr)
print(arr)
