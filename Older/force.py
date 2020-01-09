#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:32:15 2019

@author: hogbobson
"""
import numpy as np
from numpy import linalg as LA
from astropy import constants as astcnst
from scipy import constants as physcnst
# TODO: rewrite everything


def gravity(r, M):
    """ Compute gravitational acceleration at r,
    shamelessly copied from 7a in computational astrophysics. """
    rm = LA.norm(r,axis = 2)     # Find the magnitude of distances (usually SSO.d is loaded in here)
    rm[rm == 0] = np.nan    # Make 0s nan so we avoid divide by 0.
    rmcub = rm*rm*rm        # They say rm*rm*rm is faster than rm**3
    a = r/rmcub.reshape(np.append(np.shape(r[:,:,0]),1))    # This was surprisingly hard to get working
    a *= astcnst.G.value*M.reshape(1,np.size(M),1)             # So I do it one step at a time.
    a[np.isnan(a)] = 0      # Convert nans back to 0.
    acc = np.sum(a,axis=1)  # And sum all the accelerations.
    return acc*M

def electrostatic(r, M, q):
    """ Compute electrostatic forces """
    rm = LA.norm(r, axis = 2)
    rm[rm == 0] = np.nan
    rmcub = rm*rm*rm
    F = r/rmcub.reshape(np.append(np.shape(r[:,:,0]),1))    # This was surprisingly hard to get working
    F *= 1/(4*np.pi*physcnst.epsilon_0)*q*np.reshape(q, (1,len(q)))
    F[np.isnan(a)] = 0
    F = np.sum(F,axis=1)
    return F

def lennard_jones(r, **kwargs):
    pass
    
    

def spring():
    pass


def check_force(force_to_check):
    # TODO: MAKE more GENERAL
    gravity_return = {'keys': ['force gravity', 'args gravity'],
                      'fargs': (r, M)}
    electrostatic_return = {'keys': ['force electrostatic', 
                                      'args electrostatic'],
                      'fargs': (r, M, q)}
    
    if force_to_check == force.gravity:
        return gravity_return
    elif force_to_check == force.electrostatic:
        return electrostatic_return
    else:
        return False