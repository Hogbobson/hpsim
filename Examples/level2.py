#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:33:02 2019

@author: hogbobson
"""
from hpsim import main
from hpsim import visual
from hpsim import ensgen
from matplotlib import pyplot as plt
from hpsim import timegen

plt.close("all")

e = main.main(ensemble_generator = ensgen.solar_system, 
                    plot_func = visual.simple_2d_anim,
                    time_end = timegen.time_years(1))


