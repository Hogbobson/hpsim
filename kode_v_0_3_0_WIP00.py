# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:52:17 2019

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










################################################
#####_____#####     Examples     #####_____#####
################################################

#####_____#####     Ensemble Generators     #####_____#####
def star_system_functions():
    
    semi_major_axis = []
    r               = np.empty((0,3), float)
    r_magnitude     = []
    r_legacy        = np.empty((0,3,0), float)
    velocity        = []
    velocity_magnitude = []
    distance        = []
    eccentricity    = []
    inclination     = []
    mass            = []
    number_of_objects = 0
    reference_body  = np.empty(0,int)
    label           = []
    
    others   = {'sma': semi_major_axis,
                'e': eccentricity,
                'i': inclination,
                'ref': reference_body
            }
    
    ensemble = {'r': r,
                'rm': r_magnitude,
                'rdata': r_legacy,
                'd': distance,
                'v': velocity,
                'vm': velocity_magnitude,
                'm': mass,
                'n': number_of_objects,
                'label': label,
                'rem': others
            }
    
    def central_star(ensemble, mass = astcnst.M_sun.value):
        ensemble['r']          = np.append(ensemble['r'], \
                                [[0., 0., 0.]], axis = 0)
        ensemble['m']       = np.append(ensemble['m'], mass)
        ensemble['label'].append('Sun')
        ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], 0)
        ensemble['rem']['e']   = np.append(ensemble['rem']['e'], 0)
        ensemble['rem']['i']   = np.append(ensemble['rem']['i'], 0)
        ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
        return ensemble
    
    def known_planets(ensemble = None):
        """'planets' must be a dictionary containing the all the 
        wanted planets"""
        
        E_mass = astcnst.M_earth.value
        
        def mercury(ensemble):
            semi_major_axis = 57.91e9
            aphelion        = 69.82e9
            eccentricity    = 0.2056
            inclination     = rad(7.0)
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass*0.0553)
            ensemble['label'].append('Mercury')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def venus(ensemble):
            semi_major_axis = 108.21e9   # Venus Semi-major axis, [m].
            aphelion        = 108.94e9   # Venus _aphelion, [m].
            eccentricity    = 0.0067     # Venus _eccentricity
            inclination     = rad(3.39)  # Venus _inclination
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                E_mass*0.815)
            ensemble['label'].append('Venus')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
    
        def earth(ensemble):
            semi_major_axis = 149.60e9    # Earth Semi-major axis, [m].
            aphelion        = 152.1e9     # Earth _aphelion, [m].
            eccentricity    = 0.0167      # Earth _eccentricity.
            inclination     = rad(0.00005)# Earth _inclination.
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass)
            ensemble['label'].append('Earth')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def mars(ensemble):
            semi_major_axis = 227.92e9    # Mars Semi-major axis, [m].
            aphelion        = 249.23e9     # Mars _aphelion, [m].
            eccentricity    = 0.0935      # Mars _eccentricity.
            inclination     = rad(1.85)# Mars _inclination.
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass * 0.107)
            ensemble['label'].append('Mars')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def jupiter(ensemble):
            semi_major_axis = 778.57e9
            aphelion      = 816.62e9
            eccentricity  = 0.0489
            inclination   = rad(1.304)
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    astcnst.M_jup.value)
            ensemble['label'].append('Jupiter')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def saturn(ensemble):
            semi_major_axis = 1433.53e9
            aphelion        = 1514.50e9
            eccentricity    = 0.0565
            inclination     = rad(2.485)
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass * 95.16)
            ensemble['label'].append('Saturn')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def uranus(ensemble):
            semi_major_axis = 2872.46e9
            aphelion        = 3003.62e9
            eccentricity    = 0.0457
            inclination     = rad(0.772)
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass * 14.54)
            ensemble['label'].append('Uranus')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        def neptune(ensemble):
            semi_major_axis = 4495.06e9
            aphelion        = 4545.67e9
            eccentricity    = 0.0113
            inclination     = rad(1.769)
            ensemble['r']          = np.append(ensemble['r'], \
                                [[aphelion, 0., 0.]], axis = 0)
            ensemble['m']       = np.append(ensemble['m'], \
                                    E_mass * 17.15)
            ensemble['label'].append('Neptune')
            ensemble['rem']['sma'] = np.append(ensemble['rem']['sma'], \
                                    semi_major_axis)
            ensemble['rem']['e']   = np.append(ensemble['rem']['e'], \
                                    eccentricity)
            ensemble['rem']['i']   = np.append(ensemble['rem']['i'], \
                                    inclination)
            ensemble['rem']['ref'] = np.append(ensemble['rem']['ref'], 0)
            return ensemble
        
        return [mercury, venus, earth, mars, jupiter, saturn, \
                uranus, neptune]
        
        
        #for i in planets:
        #    ensemble = i(ensemble)
        #return ensemble
    
    def velocities_with_central_star(ensemble):
        """ Makes the rm vector and sets the velocities, 
        as per parametres. """
        n = ensemble['n']
        ensemble['d']  = distances(ensemble['r'])
        #dm             = LA.norm(ensemble['d'], axis = 2)
        ensemble['rm'] = LA.norm(ensemble['r'], axis = 1)
        ensemble['vm'] = np.zeros_like(ensemble['m'])
        ensemble['vm'][1:n] = np.sqrt(astcnst.G.value * \
                        ensemble['m'][0] * (2/ensemble['rm'][1:n] - \
                        1/ensemble['sma'][1:n]))
        alpha = np.reshape(np.sqrt(1. - ensemble['rem']['e']**2), (n, 1))
        ensemble['v']  = np.zeros_like(ensemble['r'])
        ensemble['v'][:n,1] = np.cos(ensemble['rem']['i'][:n]) * \
                        ensemble['vm'][:n]
        ensemble['v'][:n,2] = np.sin(ensemble['rem']['i'][:n]) * \
                        ensemble['vm'][:n]
        ensemble['v'][:n,:] = ensemble['v'][:n,:] * alpha[:n]
        ensemble['vm'] = LA.norm(ensemble['v'], axis = 1)
        return ensemble
    
    return {'star func': central_star,
            'planet funcs': known_planets(),
            'velocity func': velocities_with_central_star,
            'star system ensemble': ensemble}
    
def star_system_generator(ensemble, generator_dict):
    pass
    
    
    #for ith_dict in generator_list:
    #    ensemble = ith_dict['function name'](ensemble, ith_dict['arg'])
        
    #return ensemble

def solar_system():
    needed_funcs_dict = star_system_functions()
    sol_dict = {'function name': needed_funcs_dict['star func'],
                'arg': astcnst.M_sun.value
                }
    
    planet_dict = {'function name': needed_funcs_dict['planet funcs'],
                   'arg': [mercury, venus, earth, mars, jupiter, \
                           saturn, neptune]}
                   
    velocity_dict = {'function name': velocities_with_central_star,
                     'arg': None}
    
    gen_list = [sol_dict, planet_dict, velocity_dict]
    
    ensemble = star_system_ensemble(gen_list)



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
    return np.pi * 1e7

def time_years_precise(years):
    return time_days(365.256363004*years)
def time_Myr(mega_years):
    return time_years(1e6*mega_years)





############################################
#####_____#####     CORE     #####_____#####
############################################

#####_____#####     Miscellaneous Necessary Functions     #####_____#####
def sym_kick(ensemble, dt, d, forces):
    ensemble['v'] += d * dt * acceleration(ensemble, forces)

def sym_drift(ensemble, dt, c):
    ensemble['r'] += c * dt * ensemble['v']
    ensemble['d'] = distances(ensemble['r'])

def acceleration(ensemble, forces): #save this in the class?
    acc = np.zeros_like(ensemble['r'])
    for force_func in forces:
        acc += force_func(ensemble['d'], ensemble['m'])
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
                       wanted_forces = ['gravity'], \
                       time_start = 0, \
                       time_step = constant_dt(100000), \
                       time_end = time_years(100)):
    
    everything = {
            'ensemble': ensemble_generator(),
            'integrator': n_squared,
            'solver': sym2,
            'forces': wanted_forces,
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

    return everything






############################################
#####_____#####     USER     #####_____#####
############################################

data = sonic_the_hedgehog()















