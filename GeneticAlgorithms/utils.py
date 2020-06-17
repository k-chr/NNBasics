# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:40:56 2020

@author: Kamil Chrustowski
"""
import heapq
from numpy import *

"""
It does the chromosome inversion of existing genom
"""          
    
def count_bits(genome: array):
    val = count_nonzero(genome)
    return val
  
def getShuffle(collection:list)->list:
    rV = list(collection)
    random.shuffle(rV)
    return rV

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