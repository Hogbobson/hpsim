# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:36:40 2019

@author: Kildholt
"""
#preamble
import numpy as np
import re
import os
import time as tm
from matplotlib import pyplot as plt
from numpy import random, linalg as rng, la
from astropy import constants as astcnst
from scipy import constants as phycnst
from math import radians as rad






#forces
def gravity(a,b):
    return 

def electrostatic(a,b):
    return a-b

def force3(a,b):
    return a*b
#etc...
forces = {
        "gravity": gravity(),
        "electrostatic": electrostatic,
        "forceN": force3
        }

#techniques
def nsquared(c,d):
    return something #does it even make sense to have
# these as functions?

def tree_method(c,d):
    return your_mom

integrators = {
        "n^2": nsquared,
        "tree": tree_method,
        "techniqueN": somethingelse,
        }


#solvers
def sym1(e,r):
    return thesolution

def sym2(e,r):
    return pizza

def RK2(e,r):
    return wtf

def RK4(e,r):
    return somethingwrong

solvers = {
        "modified euler": sym1,
        "verlet": sym2,
        "improved euler": RK2,
        "runge-kutta": RK4
        }


#ensemble generator and a meager attempt at the nested functions Brian showed
#me.
def ensemble(n, rand_or_not):
    gen_func = rand_or_not
    def make_the_ensemble(x):
        return gen_func(n,x)
    return gen_func

#The thought is having both random and non_random depend on the same 
#variables, somehow. Then it should go in the same ensemble function.
def totally_random_ensemble(n = 1000,x):
    return rngsus_take_the_wheel

def not_so_random_ensemble(n,x):
    return no_make_it_like_this

def default_ensemble(n,x):
    return I_dont_know

def specific_ensemble(n,x):
    return hat

def solar_system():
    return yolo

#ensemble class, to be passed around
class ensemble():
    def __init__(self,integrator = n_squared, solver = sym, order = 1):
        
        


#Main function. Here stuff should happen. I wonder if it should have nested
#function definitions as well so as to avoid variable bloat.
def main(wanted_forces, wanted_integrator, wanted_solver,\
               wanted_ensemble):
    
    ensemble = wanted_ensemble
    integrator = wanted_integrator
    solver = wanted_solver
    
    #can we avoid a while/for?
    while time < time_that_should_pass:
        F = 0
        for function_n in wanted_forces:
            F += function_n #I imagine integrator will have an effect on the
                            #implementation of forces... perhaps? I know too
                            #few algorithms!
        new_pos = solver(old_pos,integrator(i_dunno))
        animate_if_that_should_happen(new_pos)  #animation is hard but so
                                                #visually pleasing!
        time += dt
    
    
    the_craziest_goddamn_plotting_routine_you_have_ever_seen()


def the_craziest_goddamn_plotting_routine_you_have_ever_seen():
    #plot it like you mean it




grav_and_el = main(forces[0:1], integrators[0], solvers[1], \
                   totally_random_ensemble))

grav_and_el(n = 100, max_t = 100)


