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
    def __init__(cls,integrator = n_squared, solver = sym, order = 1):
        cls._make_vars()
        cls._sol()
        cls._planets()
        
    
    def makeVars(cls):
    cls.sma = []                           # Semi-major axis. [m]
    cls.r   = np.empty((0,3), float)       # Distance from origo. [m]
    #if cls.numMoons > 0:
    #    cls.Msma = []                      # Semi Major Axes of moons. [m]
    #    cls.Mr   = np.empty((0,3), float)  # Distance from moons to their parent body. [m]
    cls.e   = []                           # _eccentricity
    cls.i   = []                           # _inclination
    cls.GM  = []                           # G*m, [m^3/s^2]
    cls.ref = np.empty(0, int)             # Keeps track of parent body
        
        
    def sol(cls,sol):
        """ Initializes the sun. """
        cls.sma = np.append(cls.sma,0)              
        cls.r   = np.append(cls.r,[[0.,0.,0.]],axis=0)      # Puttin' the Sun in Origin!
        cls.e   = np.append(cls.e,0)
        cls.i   = np.append(cls.i,0)
        cls.GM  = np.append(cls.GM,astcnst.GM_sun.value) # M_sun*G, m^3/s^2
        cls.ref = np.append(cls.ref,0)               # What the object orbits (reduntant for sun)
        
        
    def planets(cls):
        """ Makes planets. Only takes Mercury, Earth, Venus, Mars, and
        Jupiter right now. """
    
        mercu_semi_major_axis = 57.91e9
        mercu_aphelion      = 69.82e9
        mercu_eccentricity  = 0.2056
        mercu_inclination   = rad(7.0)
        cls.sma = np.append(cls.sma,mercu_semi_major_axis)
        cls.r   = np.append(cls.r,[[mercu_aphelion,0.,0.]],axis=0)
        cls.e   = np.append(cls.e,mercu_eccentricity)
        cls.i   = np.append(cls.i,mercu_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*0.0553)
        cls.ref = np.append(cls.ref,0)
        
        venus_semi_major_axis = 108.21e9   # Venus Semi-major axis, [m].
        venus_aphelion      = 108.94e9   # Venus _aphelion, [m].
        venus_eccentricity  = 0.0067     # Venus _eccentricity
        venus_inclination   = rad(3.39)  # Venus _inclination
        cls.sma = np.append(cls.sma,venus_semi_major_axis)
        cls.r   = np.append(cls.r,[[venus_aphelion,0.,0.]],axis=0)
        cls.e   = np.append(cls.e,venus_eccentricity)
        cls.i   = np.append(cls.i,venus_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*0.815)
        cls.ref = np.append(cls.ref,0)
    
        earth_semi_major_axis = 149.60e9    # Earth Semi-major axis, [m].
        earth_aphelion      = 152.1e9     # Earth _aphelion, [m].
        earth_eccentricity  = 0.0167      # Earth _eccentricity.
        earth_inclination   = rad(0.00005)# Earth _inclination.
        cls.sma = np.append(cls.sma,earth_semi_major_axis)
        cls.r   = np.append(cls.r,[[earth_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,earth_eccentricity)
        cls.i   = np.append(cls.i,earth_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value)
        cls.ref = np.append(cls.ref,0)
        cls.earthMoon(EMoon,earth_aphelion,cls.sma[-1])
        
        mars_semi_major_axis = 227.92e9    # Mars Semi-major axis, [m].
        mars_aphelion      = 249.23e9     # Mars _aphelion, [m].
        mars_eccentricity  = 0.0935      # Mars _eccentricity.
        mars_inclination   = rad(1.85)# Mars _inclination.
        cls.sma = np.append(cls.sma,mars_semi_major_axis)
        cls.r   = np.append(cls.r,[[mars_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,mars_eccentricity)
        cls.i   = np.append(cls.i,mars_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*0.107)
        cls.ref = np.append(cls.ref,0)
        
        jupit_semi_major_axis = 778.57e9
        jupit_aphelion      = 816.62e9
        jupit_eccentricity  = 0.0489
        jupit_inclination   = rad(1.304)
        cls.sma = np.append(cls.sma,jupit_semi_major_axis)
        cls.r   = np.append(cls.r,[[jupit_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,jupit_eccentricity)
        cls.i   = np.append(cls.i,jupit_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*317.83)
        cls.ref = np.append(cls.ref,0)
    
    def velocities(self):
        """ Makes the rm vector and sets the velocities, as per parametres. """
        nNM = np.size(cls.GM)               # Have the number of non-moon objects at the ready.
        self.d = distances(self.r)          # Figure out distance between each object.
        self.dm = magnitude(self.d,2)       # ... and the magnitude of these vectors.
        self.radius()                       # Set self.r
        self.vm  = np.zeros_like(self.GM)   # And preallocate self.v
        self.vm[1:nNM] = np.sqrt(cnst.GM_sun.value*(2/self.rm[1:nNM] - 1/self.sma[1:nNM]))  # The exact kepler speed, \
        # since I already typed in the semi-major axes.
        alpha    = np.reshape(np.sqrt(1. - self.e**2), (np.size(self.e), 1))    # Kepler correction alpha, stolen from 7a.   
        self.v = np.zeros_like(self.r)      # Preallocate self.v
        self.v[:nNM,1] = np.cos(self.i[:nNM])*self.vm[:nNM] #v_y = vm*cos(inclination) (for planets)
        self.v[:nNM,2] = np.sin(self.i[:nNM])*self.vm[:nNM] #v_z = vm*sin(inclination) (for planets)
        self.v[:nNM,:] = self.v[:nNM,:]*alpha[:nNM]         #multiply v by alpha
        if self.numMoons > 0:               # And do it all again for moons
            self.Md = distances(self.Mr)    # But make it so they're ready to orbit their host body.
            self.Mdm = magnitude(self.Md,2)
            self.Mrm = magnitude(self.Mr)
            self.Mvm = np.sqrt(self.GM[self.ref[nNM:]]*(2/self.Mrm - 1/self.Msma))
            self.v[nNM:,:] = self.v[self.ref[nNM:],:]           #give moons the same velocity as host planets
            self.v[nNM:,1] = self.v[nNM:,1] + np.cos(self.i[nNM:])*self.Mvm*alpha[nNM:]
            self.v[nNM:,2] = self.v[nNM:,2] + np.sin(self.i[nNM:])*self.Mvm*alpha[nNM:]
        self.speed()                        # Set self.vm once more
        self.m   = self.GM/cnst.G.value     # Have a mass variable, just in case.
        self.p   = self.vm*self.m           # And a momentum variable.

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


