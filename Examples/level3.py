#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:48:23 2019

@author: hogbobson
"""
from hpsim import main
from hpsim.miscfuncs import distances
from hpsim.miscfuncs import acceleration

def euler_integration(ensemble, dt, forces):
    ensemble['r'] += dt * ensemble['velocity']
    ensemble['distance'] = distances(ensemble['r'])
    ensemble['velocity'] += dt * acceleration(ensemble, forces)
    return ensemble

def RG_integration(ensemble, dt, forces):
    ensemble['r'] += 2 * dt * ensemble['velocity']
    ensemble['distance'] = distances(ensemble['r'])
    ensemble['velocity'] += dt + acceleration(ensemble, forces)
    
def four(ensemble, dt, forces):
    ensemble['r'] += 5*4e9
    return ensemble

e = main.main(solver_func = four)