#!/usr/bin/env python
# shared_classes.py - Shared standardized classes for HOF compatibility
# Author: Ou Deng on Oct 15, 2025

"""
Shared classes for standardized HOF representation across all SR methods.
These classes ensure pickle compatibility between different scripts.
"""

class StandardizedFitness:
    """Fitness class compatible with DEAP-style individuals"""
    def __init__(self, values):
        self.values = values

class StandardizedIndividual:
    """Individual class compatible with DEAP-style HOF"""
    def __init__(self, expr, fitness_values):
        self.expr = expr
        if isinstance(fitness_values, StandardizedFitness):
            self.fitness = fitness_values
        else:
            self.fitness = StandardizedFitness(fitness_values)
        
    def __str__(self):
        return str(self.expr)
    
    @property 
    def height(self):
        """Estimate tree height from expression"""
        return self._estimate_height()
    
    def _estimate_height(self):
        """Simple heuristic: count parentheses depth"""
        max_depth = 0
        current = 0
        for char in str(self.expr):
            if char == '(':
                current += 1
                max_depth = max(max_depth, current)
            elif char == ')':
                current -= 1
        return max_depth

def create_standardized_individual(expr, mse, complexity):
    """Helper function to create a standardized individual"""
    return StandardizedIndividual(expr, (mse, complexity))