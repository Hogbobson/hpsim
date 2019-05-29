# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:52:17 2019

@author: Kildholt
"""

#####_____#####     PREAMBLE     #####_____#####
import numpy as np
#import re
#import os
#import time as tm
from matplotlib import pyplot as plt
from numpy import random as rng, linalg as LA
from astropy import constants as astcnst
from scipy import constants as phycnst
from math import radians as rad
plt.close("all")










################################################
#####_____#####     Examples     #####_____#####
################################################

#####_____#####     Ensemble Generators     #####_____#####
class StarSystemGenerator:
    def __init__(cls):
        cls.sma      = []
        cls.r        = np.empty((0,3), float)
        cls.rm       = []
        cls.r_legacy = np.empty((0,3,0), float)
        cls.v        = []
        cls.vm       = []
        cls.d        = []
        cls.e        = []
        cls.i        = []
        cls.m        = []
        cls.n        = 0
        cls.ref      = np.empty(0,int)
        cls.label    = []
        
    
    def get_ensemble(cls):
        others   = {'semi major axis': cls.sma,
                    'eccentricity': cls.e,
                    'inclination': cls.i,
                    'reference body': cls.ref
                }
        
        ensemble = {'r': cls.r,
                    'r magnitude': cls.rm,
                    'r data': cls.r_legacy,
                    'distance': cls.d,
                    'velocity': cls.v,
                    'velocity magnitude': cls.vm,
                    'mass': cls.m,
                    'number of objects': cls.n,
                    'label': cls.label,
                    'rem': others
                }
        return ensemble
    
    
    def set_ensemble(cls, ensemble):
        cls.sma      = ensemble['rem']['semi major axis']
        cls.r        = ensemble['r']
        cls.rm       = ensemble['r magnitude']
        cls.r_legacy = ensemble['r data']
        cls.v        = ensemble['velocity']
        cls.vm       = ensemble['velocity magnitude']
        cls.d        = ensemble['distance']
        cls.e        = ensemble['rem']['eccentricity']    
        cls.i        = ensemble['rem']['inclination']
        cls.m        = ensemble['mass']
        cls.n        = ensemble['number of objects']
        cls.ref      = ensemble['rem']['reference body']
        cls.label    = ensemble['label']
        
        
        
    def central_star(cls, mass = astcnst.M_sun.value):
        cls.r          = np.append(cls.r, \
                                [[0., 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, mass)
        cls.label.append('Sun')
        cls.sma = np.append(cls.sma, 0)
        cls.e   = np.append(cls.e, 0)
        cls.i   = np.append(cls.i, 0)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
    
    ##### KNOWN PLANETS #####
    def mercury(cls):
        semi_major_axis = 57.91e9
        aphelion        = 69.82e9
        eccentricity    = 0.2056
        inclination     = rad(7.0)
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value*0.0553)
        cls.label.append('Mercury')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
    
    def venus(cls):
        semi_major_axis = 108.21e9   # Venus Semi-major axis, [m].
        aphelion        = 108.94e9   # Venus _aphelion, [m].
        eccentricity    = 0.0067     # Venus _eccentricity
        inclination     = rad(3.39)  # Venus _inclination
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                            astcnst.M_earth.value*0.815)
        cls.label.append('Venus')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        

    def earth(cls):
        semi_major_axis = 149.60e9    # Earth Semi-major axis, [m].
        aphelion        = 152.1e9     # Earth _aphelion, [m].
        eccentricity    = 0.0167      # Earth _eccentricity.
        inclination     = rad(0.00005)# Earth _inclination.
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value)
        cls.label.append('Earth')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        
    
    def mars(cls):
        semi_major_axis = 227.92e9    # Mars Semi-major axis, [m].
        aphelion        = 249.23e9     # Mars _aphelion, [m].
        eccentricity    = 0.0935      # Mars _eccentricity.
        inclination     = rad(1.85)# Mars _inclination.
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value * 0.107)
        cls.label.append('Mars')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        
    
    def jupiter(cls):
        semi_major_axis = 778.57e9
        aphelion      = 816.62e9
        eccentricity  = 0.0489
        inclination   = rad(1.304)
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_jup.value)
        cls.label.append('Jupiter')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        
    
    def saturn(cls):
        semi_major_axis = 1433.53e9
        aphelion        = 1514.50e9
        eccentricity    = 0.0565
        inclination     = rad(2.485)
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value * 95.16)
        cls.label.append('Saturn')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        
    
    def uranus(cls):
        semi_major_axis = 2872.46e9
        aphelion        = 3003.62e9
        eccentricity    = 0.0457
        inclination     = rad(0.772)
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value * 14.54)
        cls.label.append('Uranus')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
        
    
    def neptune(cls):
        semi_major_axis = 4495.06e9
        aphelion        = 4545.67e9
        eccentricity    = 0.0113
        inclination     = rad(1.769)
        cls.r          = np.append(cls.r, \
                            [[aphelion, 0., 0.]], axis = 0)
        cls.m       = np.append(cls.m, \
                                astcnst.M_earth.value * 17.15)
        cls.label.append('Neptune')
        cls.sma = np.append(cls.sma, \
                                semi_major_axis)
        cls.e   = np.append(cls.e, \
                                eccentricity)
        cls.i   = np.append(cls.i, \
                                inclination)
        cls.ref = np.append(cls.ref, 0)
        cls.n  += 1
    ##### END KNOWN PLANETS #####
    
    def all_known_planets(cls):
        cls.mercury()
        cls.venus()
        cls.earth()
        cls.mars()
        cls.jupiter()
        cls.saturn()
        cls.uranus()
        cls.neptune()
        
    def random_planets(cls, num_ran_planets):
        def log_uniform(low = 1, high = 2, size = 1, base = 10):
            return np.power(base, rng.uniform(low, high, size))
        
        for i in range(num_ran_planets):
            x = log_uniform(9, 11)
            y = log_uniform(9, 11)
            z = log_uniform(0, 7)
            r = np.reshape(np.array([x, y, z]), (1,3))
            m = np.maximum(rng.randn()+3, 1e-5)*astcnst.M_earth.value*1e-3
            rmax = LA.norm(np.array([x, y, z]), axis = 0)
            e = np.minimum(np.abs(rng.randn()*0.1), 0.95)
            sma = rmax / (1 + e)
            i = np.arccos(z/rmax)
            cls.r = np.append(cls.r, r, axis = 0)
            cls.m = np.append(cls.m, m)
            cls.label.append('random planet #' + str(i))
            cls.sma = np.append(cls.sma, sma)
            cls.e = np.append(cls.e, e)
            cls.i = np.append(cls.i, i)
            cls.ref = np.append(cls.ref, 0)
            cls.n += 1
                    
    
    def velocities_with_central_star(cls):
        """ Makes the rm vector and sets the velocities, 
        as per parametres. """
        n = cls.n
        cls.d  = distances(cls.r)
        #dm             = LA.norm(cls.d, axis = 2)
        cls.rm = LA.norm(cls.r, axis = 1)
        theta_v = np.zeros(n)
        theta_v[1:n] = np.pi/2 - np.arctan(np.abs(cls.r[1:n, 1]/cls.r[1:n, 0]))
        cls.vm = np.zeros_like(cls.m)
        cls.vm[1:n] = np.sqrt(astcnst.G.value * \
                        cls.m[0] * (2/cls.rm[1:n] - \
                        1/cls.sma[1:n]))
        alpha = np.reshape(np.sqrt(1. - cls.e**2), (n, 1))
        cls.v  = np.zeros_like(cls.r)
        cls.v[:n,0] = -1 * np.sign(cls.r[:n,1]) * \
                        np.cos(theta_v[:n]) * cls.vm[:n]
        cls.v[:n,1] = np.sign(cls.r[:n,0]) * \
                        np.cos(theta_v[:n]) * cls.vm[:n]
        cls.v[:n,:] = cls.v[:n,:] * alpha[:n]
        cls.v[:n,2] = cls.vm - LA.norm(cls.v[:n,:], axis = 1)
        cls.vm = LA.norm(cls.v, axis = 1)
        cls.r_legacy = np.reshape(cls.r, (n, 3, 1))
    
    
    
    #for ith_dict in generator_list:
    #    ensemble = ith_dict['function name(ensemble, ith_dict['arg)
        
    #return ensemble

def solar_system():
    SSG = StarSystemGenerator()
    SSG.central_star()
    SSG.all_known_planets()
    SSG.velocities_with_central_star()
    
    ensemble = SSG.get_ensemble()
    return ensemble
    
def random_solar_system():
    SSG = StarSystemGenerator()
    SSG.central_star()
    SSG.random_planets(10)
    SSG.velocities_with_central_star()
    
    ensemble = SSG.get_ensemble()
    return ensemble
    


#####_____#####     Integrators     #####_____#####
def n_squared(ensemble):
    pass



#####_____#####     Solvers     #####_____#####
def sym1(ensemble, dt, forces):
    ensemble = sym_kick(ensemble, dt, 1, forces)
    ensemble = sym_drift(ensemble, dt, 1)
    return ensemble
    
def sym2(ensemble, dt, forces):
    ensemble = sym_kick(ensemble, dt, 0.5, forces)
    ensemble = sym_drift(ensemble, dt, 1)
    ensemble = sym_kick(ensemble, dt, 0.5, forces)
    return ensemble

def sym3(ensemble, dt, forces):
    pass

#Figure out how to make these two v^ not awful.

def sym4(ensemble, dt, forces):
    pass



#####_____#####     Force Functions     #####_____#####
def gravity(r,M):
    """ Compute gravitational acceleration at r,
    shamelessly copied from 7a in computational astrophysics. """
    rm = LA.norm(r,axis = 2)     # Find the magnitude of distances (usually SSO.d is loaded in here)
    rm[rm == 0] = np.nan    # Make 0s nan so we avoid divide by 0.
    rmcub = rm*rm*rm        # They say rm*rm*rm is faster than rm**3
    a = r/rmcub.reshape(np.append(np.shape(r[:,:,0]),1))    # This was surprisingly hard to get working
    a *= astcnst.G.value*M.reshape(1,np.size(M),1)             # So I do it one step at a time.
    a[np.isnan(a)] = 0      # Convert nans back to 0.
    acc = np.sum(a,axis=1)  # And sum all the accelerations.
    return acc

def electrostatic():
    pass

def lennard_jones():
    pass

def spring():
    pass





#####_____#####     Time Step Functions     #####_____#####
def constant_dt(value):
    return value

def courant():
    pass




#####_____#####     Time Generators     #####_____#####
def time_seconds(seconds):
    return seconds

def time_minutes(minutes):
    return time_seconds(60*minutes)

def time_hours(hours):
    return time_minutes(60*hours)

def time_days(days):
    return time_hours(24*days)

def time_months(months):
    return time_days(30*months)

def time_years(years): #You cant top the elegance of pi*1e7!
    return np.pi * 1e7 * years

def time_years_precise(years):
    return time_days(365.256363004*years)

def time_Myr(mega_years):
    return time_years(1e6*mega_years)



# =============================================================================
# ensemble = {'r': cls.r,
#                     'r magnitude': cls.rm,
#                     'r data': cls.r_legacy,
#                     'distance': cls.d,
#                     'velocity': cls.v,
#                     'velocity magnitude': cls.vm,
#                     'mass': cls.m,
#                     'number of objects': cls.n,
#                     'label': cls.label,
#                     'rem': others
#                 }
# =============================================================================
        

#####_____#####     Plotting Functions     #####_____#####
def no_plot(ensemble):
    pass

def standard_plot(ensemble):
    n = ensemble['number of objects']
    plt.figure(num = 0, figsize = (8,8))
    for i in range(n):
        plt.plot(ensemble['r data'][i,0,:], ensemble['r data'][i,1,:], '-', \
                 label = ensemble['label'][i], \
                 color = (abs(np.sin(i)), 0 + i/n, 1-i/n))
    plt.xlabel('x')
    plt.ylabel('y')
    xlim = np.max(ensemble['r data'])*1.1
    xlim = 5*1e11
    plt.axis((-xlim, xlim, -xlim, xlim))
    plt.legend()
    plt.show()


############################################
#####_____#####     CORE     #####_____#####
############################################

#####_____#####     Miscellaneous Necessary Functions     #####_____#####
def sym_kick(ensemble, dt, d, forces):#, acceleration):
    ensemble['velocity'] += d * dt * acceleration(ensemble, forces)
    return ensemble

def sym_drift(ensemble, dt, c):
    ensemble['r'] += c * dt * ensemble['velocity']
    ensemble['distance'] = distances(ensemble['r'])
    return ensemble

def acceleration(ensemble, forces): #save this in the class?
    acc = np.zeros_like(ensemble['r'])
    for force_func in forces:
        acc += force_func(ensemble['distance'], ensemble['mass'])
    return acc

def distances(vec): # There must be an easier way.
    """ Converts matrix elements from origin -> object to object -> object. \
    Naturally, the dimensions in the matrix increase because of that. """
    newr = np.zeros((np.shape(vec)[0],np.shape(vec)[0],3))
    for i in range(3):  #For future: get rid of loop, if possible.
        newr[:,:,i] = vec[:,i].reshape(1,np.size(vec[:,i])) - \
        vec[:,i].reshape(np.size(vec[:,i]),1)
    return newr 



#####_____#####     MAIN FUNCTION NAMED AFTER SONIC      #####_____#####
#####_____#####     BECAUSE IT'S SUPPOSED TO BE FAST     #####_____#####
def sonic_the_hedgehog(ensemble_generator = solar_system, \
                       integration_func = n_squared, \
                       solver_func = sym2, \
                       wanted_forces = [gravity], \
                       plot_func = standard_plot, \
                       time_start = 0, \
                       time_step = constant_dt(100000), \
                       time_end = time_years(1)):
    
    everything = {
            'ensemble': ensemble_generator(),
            'integrator': integration_func,
            'solver': solver_func,
            'forces': wanted_forces,
            'plotter': plot_func,
            'time start': time_start,
            'current time': time_start,
            'dt': time_step,
            'time end': time_end
            }
    
    #I put everything here that isn't changed for readability.
    integrator = everything['integrator']
    solver     = everything['solver']
    forces     = everything['forces']
    
    while everything['current time'] < everything['time end']:
        integrator(everything['ensemble'])
        solver(everything['ensemble'], everything['dt'], forces)
        everything['current time'] += everything['dt']
        everything['ensemble']['r data'] = np.append( \
                  everything['ensemble']['r data'], 
                  np.reshape(everything['ensemble']['r'], \
                        (everything['ensemble']['number of objects'],3,1)), \
                             axis = 2)
    
    plot_func(everything['ensemble'])

    return everything






############################################
#####_____#####     USER     #####_____#####
############################################

data = sonic_the_hedgehog(ensemble_generator = random_solar_system)















