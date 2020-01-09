#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main function of hpsim. Everything should (be able to be) run from
here. The main function takes the following arguments, with several default op-
tions:
    arg                 default                 other options
    ensemble_generator  ensgen.solar_system     ensgen.random_solar_system
    integration_func    ntgrtr.n_squared
    solver_func         solver.sym2
    wanted_forces       [force.gravity]
    plot_func           visual.simple_2d_anim   visual.no_plot, 
                                                visual.simple_plot
    plot_save           False
    energy_func         energy.no_energy
    time_start          0                       <any number (float or int)>
    time_step           timestep.constant_dt(1e5)   <other values in the arg>
    time_end            timegen.time_years(1)   <other values in the arg>
                                                timegen.time_years_precise(num)
                                                timegen.time_Myr(mega_years)
                                                timegen.time_months(months)
                                                timegen.time_days(days)
                                                timegen.time_hours(hours)
                                                timegen.time_minutes(minutes)
                                                timegen.time_seconds(seconds)
    
For each of these inputs, it is possible for the user (read: you) to create his
or her own functions and load them in. Be advised that there are as of yet not
checks for every function to see if the user input is correct. The default ca-
ses work, but the program might break with something new. I am working on rec-
tifying that particular issue.
"""

import numpy as np
from . import force
from . import miscfuncs
from . import solver
from . import ntgrtr
from . import ensgen
from . import timestep
from . import timegen
from . import visual
from . import energy



def main(ensemble_generator = ensgen.solar_system,
         integration_func = ntgrtr.n_squared, 
         solver_func = solver.sym2,
         wanted_forces = [force.gravity],
         plot_func = visual.simple_2d_anim,
         plot_save = False,
         energy_func = energy.no_energy,   
         time_start = 0, 
         time_step = timestep.constant_dt(100000), 
         time_end = timegen.time_years(1)
                   ):
    
    
    # Get the strings (dictionary keys) associated with the wanted forces.
    force_variables = miscfuncs.get_force_variables(wanted_forces)


    # This dict gathers all of the information given to main. A dict is used as
    # opposed to a class because my main supoervisor hates OOP. If you are 
    # main supervisor: 1: Hi Brian!, 2: Wouldn't a class be slightly easier
    # here? I think it could do good things.
    everything = {
        'ensemble': ensemble_generator(force_variables),
        'integrator': integration_func,
        'solver': solver_func,
        'forces': wanted_forces,
        'force variables': force_variables,
        'plotter': plot_func,
        'time start': time_start,
        'current time': time_start,
        'dt': time_step,
        'time end': time_end,
        'time steps': 0,
        'plot output': None
            }
    
    # This function checks the generated ensemble.
    miscfuncs.ensemble_checker(everything['ensemble'], \
                               everything['force variables'])
    
    
    #I put everything here that isn't changed over the course of the loop
    # for readability.
    integrator = everything['integrator']
    solver     = everything['solver']
    forces     = everything['forces']
    
    # The thought of this loop is as follows:
    # while the current time < the designated end time:
    #    check the integration method (n^2, PIC, tree, etc.)
    #    make one time step happen
    #    add dt to the current time
    #    add 1 to the step counter
    #    append position data to the position data tensor
    while everything['current time'] < everything['time end']:
        integrator(everything['ensemble'])
        solver(everything['ensemble'], everything['dt'], forces)
        everything['current time'] += everything['dt']
        everything['time steps'] += 1
        everything['ensemble']['r data'] = np.append( \
                  everything['ensemble']['r data'], 
                  np.reshape(everything['ensemble']['r'], \
                        (everything['ensemble']['number of objects'], \
                         3,1)), axis = 2)
    
    # Plot me baby, one more time.
    everything['plot output'] = plot_func(everything)
    
    # It is useful to output the dict.
    return everything