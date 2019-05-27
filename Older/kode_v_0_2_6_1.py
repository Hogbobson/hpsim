# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:36:40 2019

@author: Kildholt
"""

#####_____#####     PREAMBLE     #####_____#####
import numpy as np
import re
import os
import time as tm
from matplotlib import pyplot as plt
from numpy import random as rng, linalg as LA
from astropy import constants as astcnst
from scipy import constants as phycnst
from math import radians as rad
plt.close("all")





#####_____#####     FORCES     #####_____#####
def gravity(r,GM):
    """ Compute gravitational acceleration at r,
    shamelessly copied from 7a in computational astrophysics. """
    rm = LA.norm(r,axis = 2)     # Find the magnitude of distances (usually SSO.d is loaded in here)
    rm[rm == 0] = np.nan    # Make 0s nan so we avoid divide by 0.
    rmcub = rm*rm*rm        # They say rm*rm*rm is faster than rm**3
    a = r/rmcub.reshape(np.append(np.shape(r[:,:,0]),1))    # This was surprisingly hard to get working
    a = a*GM.reshape(1,np.size(GM),1)             # So I do it one step at a time.
    a[np.isnan(a)] = 0      # Convert nans back to 0.
    acc = np.sum(a,axis=1)  # And sum all the accelerations.
    return acc

def electrostatic(a,b):
    acc = np.zeros_like(a)
    return acc

def force3(a,b):
    return a*b
#etc...
forces = {
        "gravity": gravity,
        "electrostatic": electrostatic,
        "forceN": force3
        }





#####_____#####     INTEGRATORS     #####_____#####
def nsquared(ensemble):
    return ensemble

def tree_method(c,d):
    return your_mom

integrators = {
        "n^2": nsquared,
        "tree": tree_method,
        #"techniqueN": somethingelse,
        }






#####_____#####     SOLVERS     #####_____#####
def sym():
    def drift(ensemble, dt, c): #I wish to make these more general
        ensemble.r = ensemble.r + c * dt * ensemble.v
        ensemble.d = distances(ensemble.r)
    
    def kick(ensemble, dt, d):
        ensemble.v = ensemble.v + d * dt * acceleration(ensemble)
    
    return drift, kick

def sym_drift():
    pass

def sym_kick():
    pass


def sym1():
    
    drift, kick = sym()
    
    def iterate(ensemble,dt):
        kick(ensemble, dt, 1)
        drift(ensemble, dt, 1)
    
    return iterate


def sym2():
    
    drift, kick = sym()
    
    def iterate(ensemble, dt):
        kick(ensemble, dt, 0.5)
        drift(ensemble, dt, 1)
        kick(ensemble, dt, 0.5)
    
    return iterate

def sym22(ensamble, dt,i):
    sym_kick()
    sym_drift()
    


def sym3():
    
    drift, kick = sym()
    d = [-1/24, 3/4, 7/24]
    c = [1, -2/3, 2/3]
    
    def iterate(ensemble, dt):
        for i in range(len(d)):
            kick(ensemble, dt, d[i])
            drift(ensemble, dt, c[i])
    
    return iterate


def sym4():
    
    drift, kick = sym()
    q = 2**(1/3)
    p = 2*(2 - q)
    d = [1/(2-q), -q/(2-q), 1/(2-q), 0]
    c = [1/p, (1-q)/p, (1-q)/p, 1/p]
    
    def iterate(ensemble, dt):
        for i in range(len(d)):
            kick(ensemble, dt, d[i])
            drift(ensemble, dt, c[i])
    
    return iterate

def RK2(e,r):
    return wtf

def RK4(e,r):
    return somethingwrong

solvers = {
        "modified euler": sym1,
        "sym1": sym1,
        "verlet": sym2,
        "leapfrog": sym2,
        "sym2": sym2,
        "sym3": sym3,
        "sym4": sym4,
        "improved euler": RK2,
        "runge-kutta": RK4
        }





#####_____#####     FUNCTIONS I DON'T KNOW WHERE TO PUT     #####_____#####
def acceleration(ensemble): #save this in the class?
    acc = np.zeros_like(ensemble.r)
    for force_key in ensemble.wanted_forces:
        force_func = forces[force_key]
        acc += force_func(ensemble.d, ensemble.GM)
    return acc

def distances(vec): # There must be an easier way.
    """ Converts matrix elements from origin -> object to object -> object. \
    Naturally, the dimensions in the matrix increase because of that. """
    newr = np.zeros((np.shape(vec)[0],np.shape(vec)[0],3))
    for i in range(3):  #For future: get rid of loop, if possible.
        newr[:,:,i] = vec[:,i].reshape(1,np.size(vec[:,i])) - \
        vec[:,i].reshape(np.size(vec[:,i]),1)
    return newr 





#####_____#####     ENSEMBLE CLASS     #####_____#####
class Ensemble():
    def __init__(cls, integrator = 'n^2', solver = sym22, order = 1, \
                 num_steps = 100, wanted_forces = ['gravity']):
        cls._make_vars()
        cls._sol()
        cls._planets()
        cls._velocities()
        cls.integration_method = integrator
        cls.solving_method     = solver
        cls.wanted_forces      = wanted_forces
        cls.order              = order
        cls.num_steps          = num_steps
        cls.r_legacy           = np.reshape(cls.r_legacy, \
                                            ((np.size(cls.GM),3,0)))
        cls.r_legacy           = np.append(cls.r_legacy, \
                                 np.reshape(cls.r,(np.size(cls.GM),3,1)), \
                                 axis = 2)
        
    
    def _make_vars(cls):
        cls.sma = []                           # Semi-major axis. [m]
        cls.r   = np.empty((0,3), float)       # Distance from origo. [m]
        cls.r_legacy = np.empty((0,3,0), float)
        #if cls.numMoons > 0:
        #    cls.Msma = []                      # Semi Major Axes of moons. [m]
        #    cls.Mr   = np.empty((0,3), float)  # Distance from moons to their parent body. [m]
        cls.e   = []                           # _eccentricity
        cls.i   = []                           # _inclination
        cls.GM  = []                           # G*m, [m^3/s^2]
        cls.ref = np.empty(0, int)             # Keeps track of parent body
        cls.label = []
        
        
    def _sol(cls):
        """ Initializes the sun. """
        cls.sma = np.append(cls.sma,0)              
        cls.r   = np.append(cls.r,[[0.,0.,0.]],axis=0)      # Puttin' the Sun in Origin!
        cls.e   = np.append(cls.e,0)
        cls.i   = np.append(cls.i,0)
        cls.GM  = np.append(cls.GM,astcnst.GM_sun.value) # M_sun*G, m^3/s^2
        cls.ref = np.append(cls.ref,0)               # What the object orbits (reduntant for sun)
        cls.label.append('Sun')
        
        
    def _planets(cls):
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
        cls.label.append('Mercury')
        
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
        cls.label.append('Venus')
    
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
        cls.label.append('Earth')
        #cls.earthMoon(EMoon,earth_aphelion,cls.sma[-1])
        
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
        cls.label.append('Mars')
        
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
        cls.label.append('Jupiter')
        
        satur_semi_major_axis = 1433.53e9
        satur_aphelion        = 1514.50e9
        satur_eccentricity    = 0.0565
        satur_inclination     = rad(2.485)
        cls.sma = np.append(cls.sma,satur_semi_major_axis)
        cls.r   = np.append(cls.r,[[satur_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,satur_eccentricity)
        cls.i   = np.append(cls.i,satur_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*95.16)
        cls.ref = np.append(cls.ref,0)
        cls.label.append('Saturn')
        
        uranu_semi_major_axis = 2872.46e9
        uranu_aphelion        = 3003.62e9
        uranu_eccentricity    = 0.0457
        uranu_inclination     = rad(0.772)
        cls.sma = np.append(cls.sma,uranu_semi_major_axis)
        cls.r   = np.append(cls.r,[[uranu_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,uranu_eccentricity)
        cls.i   = np.append(cls.i,uranu_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*14.54)
        cls.ref = np.append(cls.ref,0)
        cls.label.append('Uranus')
        
        neptu_semi_major_axis = 4495.06e9
        neptu_aphelion        = 4545.67e9
        neptu_eccentricity    = 0.0113
        neptu_inclination     = rad(1.769)
        cls.sma = np.append(cls.sma,neptu_semi_major_axis)
        cls.r   = np.append(cls.r,[[neptu_aphelion,0.,0.]],axis=0) #Put all objects in x = _aphelion
        cls.e   = np.append(cls.e,neptu_eccentricity)
        cls.i   = np.append(cls.i,neptu_inclination)
        cls.GM  = np.append(cls.GM,astcnst.GM_earth.value*17.15)
        cls.ref = np.append(cls.ref,0)
        cls.label.append('Neptune')
        
        
    
    def _velocities(cls):
        """ Makes the rm vector and sets the velocities, as per parametres. """
        nNM = np.size(cls.GM)               # Have the number of non-moon objects at the ready.
        cls.d = distances(cls.r)          # Figure out distance between each object.
        cls.dm = LA.norm(cls.d,axis = 2)       # ... and the magnitude of these vectors.
        cls.radius()                       # Set cls.r
        cls.vm  = np.zeros_like(cls.GM)   # And preallocate cls.v
        cls.vm[1:nNM] = np.sqrt(astcnst.GM_sun.value*(2/cls.rm[1:nNM] - \
              1/cls.sma[1:nNM]))  # The exact kepler speed, \
        # since I already typed in the semi-major axes.
        alpha = np.reshape(np.sqrt(1. - cls.e**2), (np.size(cls.e), 1))    # Kepler correction alpha, stolen from 7a.   
        cls.v = np.zeros_like(cls.r)      # Preallocate cls.v
        cls.v[:nNM,1] = np.cos(cls.i[:nNM])*cls.vm[:nNM] #v_y = vm*cos(inclination) (for planets)
        cls.v[:nNM,2] = np.sin(cls.i[:nNM])*cls.vm[:nNM] #v_z = vm*sin(inclination) (for planets)
        cls.v[:nNM,:] = cls.v[:nNM,:]*alpha[:nNM]         #multiply v by alpha
        #if cls.numMoons > 0:               # And do it all again for moons
        #    cls.Md = distances(cls.Mr)    # But make it so they're ready to orbit their host body.
        #    cls.Mdm = magnitude(cls.Md,2)
        #    cls.Mrm = magnitude(cls.Mr)
        #    cls.Mvm = np.sqrt(cls.GM[cls.ref[nNM:]]*(2/cls.Mrm - 1/cls.Msma))
        #    cls.v[nNM:,:] = cls.v[cls.ref[nNM:],:]           #give moons the same velocity as host planets
        #    cls.v[nNM:,1] = cls.v[nNM:,1] + np.cos(cls.i[nNM:])*cls.Mvm*alpha[nNM:]
        #    cls.v[nNM:,2] = cls.v[nNM:,2] + np.sin(cls.i[nNM:])*cls.Mvm*alpha[nNM:]
        cls.speed()                        # Set cls.vm once more
        cls.m   = cls.GM/astcnst.G.value     # Have a mass variable, just in case.
        cls.p   = cls.vm*cls.m           # And a momentum variable.

    def radius(cls):
        cls.rm = LA.norm(cls.r, axis = 1)
    
    def speed(cls):
        cls.vm = LA.norm(cls.v, axis = 1)





#####_____#####     PLOTTING     #####_____#####
def this_function_makes_me_plot_hard(ensemble):
    ran = int(np.shape(ensemble.r)[0])
    plt.figure(num = 0, figsize = (8,8))
    for i in range(ran):
        plt.plot(ensemble.r_legacy[i,0,:], ensemble.r_legacy[i,1,:], '-', \
                 label = ensemble.label[i], \
                 color = (abs(np.sin(i)), 0 + i/ran, 1-i/ran))
    plt.xlabel('x')
    plt.ylabel('y')
    xlim = np.max(ensemble.r_legacy)*1.1
    plt.axis((-xlim, xlim, -xlim, xlim))
    plt.legend()
    plt.show()
        




#####_____#####     MAIN FUNCTION NAMED AFTER SONIC      #####_____#####
#####_____#####     BECAUSE IT'S SUPPOSED TO BE FAST     #####_____#####
def sonic_the_hedgehog(ensemble): #what the fuck do you call the run func?
    stps = 0
    dt = 1000000
    integration_func = integrators[ensemble.integration_method]
    solver_name = ensemble.solving_method + str(ensemble.order)
    solver_func = solvers[solver_name]
    iterator    = solver_func()
    while stps < ensemble.num_steps:
        integration_func(ensemble)
        solver(ensemble,dt, stps)
        ensemble.r_legacy = np.append(ensemble.r_legacy, \
                            np.reshape(ensemble.r,(np.size(ensemble.GM),3,1)),\
                            axis = 2)
        stps += 1
    this_function_makes_me_plot_hard(ensemble)
    return ensemble





#####_____#####     GO     #####_____#####
def tt(e,st,i):
    e.hej += 2    

SS = Ensemble(num_steps = 2000, order = 4, solver=tt)
SS = sonic_the_hedgehog(SS)









#####_____#####     GARBAGE     #####_____#####
# =============================================================================
# #Main function. Here stuff should happen. I wonder if it should have nested
# #function definitions as well so as to avoid variable bloat.
# def main(wanted_forces, wanted_integrator, wanted_solver,\
#                wanted_ensemble):
#     
#     ensemble = wanted_ensemble
#     integrator = wanted_integrator
#     solver = wanted_solver
#     
#     #can we avoid a while/for?
#     while time < time_that_should_pass:
#         F = 0
#         for function_n in wanted_forces:
#             F += function_n #I imagine integrator will have an effect on the
#                             #implementation of forces... perhaps? I know too
#                             #few algorithms!
#         new_pos = solver(old_pos,integrator(i_dunno))
#         animate_if_that_should_happen(new_pos)  #animation is hard but so
#                                                 #visually pleasing!
#         time += dt
#     
#     
#     the_craziest_goddamn_plotting_routine_you_have_ever_seen()
# 
# 
# def the_craziest_goddamn_plotting_routine_you_have_ever_seen():
#     a = a#plot it like you mean it
# =============================================================================


# =============================================================================
# def apply_integrator(ensemble):
#     "Deprecated - it didn't make sense to do this"
#     
#     return ensemble
# =============================================================================
    
# =============================================================================
# def apply_solver(ensemble):
#     "Deprecated - it didn't make sense to do this"
#     
#     
#     return ensemble
# =============================================================================


# =============================================================================
# #ensemble generator and a meager attempt at the nested functions Brian showed
# #me.
# def ensemble(n, rand_or_not):
#     gen_func = rand_or_not
#     def make_the_ensemble(x):
#         return gen_func(n,x)
#     return gen_func
# 
# #The thought is having both random and non_random depend on the same 
# #variables, somehow. Then it should go in the same ensemble function.
# def totally_random_ensemble(x, n = 1000):
#     return rngsus_take_the_wheel
# 
# def not_so_random_ensemble(n,x):
#     return no_make_it_like_this
# 
# def default_ensemble(n,x):
#     return I_dont_know
# 
# def specific_ensemble(n,x):
#     return hat
# 
# def solar_system():
#     return yolo
# =============================================================================




#grav_and_el = main(forces[0:1], integrators[0], solvers[1], \
#                   totally_random_ensemble))

#grav_and_el(n = 100, max_t = 100)


