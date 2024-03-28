#########################################################
# Copyright 2024 Haonan Zheng (nan.h.zheng@outlook.com) #
# Our model is detailed in the appendix of our paper:   #
# https://arxiv.org/abs/2403.17044,                     #
# it would be great if you find it useful and cite it!  #
#########################################################

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
plt.style.use('classic')

import matplotlib as mpl
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
parameters = {'figure.figsize': (12, 4.5), 'font.size': 15, 'axes.labelsize': 18, 'xtick.major.pad': 6, \
              'xtick.major.size': 6, 'xtick.minor.size': 4, "xtick.major.width": 0.8, "xtick.minor.width": 0.8, \
              'ytick.major.size': 6, 'ytick.minor.size': 4, "ytick.major.width": 0.8, "ytick.minor.width": 0.8}
plt.rcParams.update(parameters)

class model_fbar_NFW:
    def __init__(self, z=0, Omega_m=0.307, h=0.6777, Ti=245., zi=127., mu=50./41., massdef='mean200'):

        self.G = 6.674e-11       # gravity constant, in m^3/kg/s^2
        self.Msun = 1.989e30     # solar mass, in kg
        self.Mpc = 3.085678e22   # Mpc, in meters
        self.kB = 1.38064852e-23 # Boltzmann constant, in m^2 kg s^-2 K^-1
        self.mp = 1.6726219e-27  # proton mass, in kg

        self.z = z
        self.Omega_m = Omega_m
        self.Omega_L = 1 - Omega_m
        self.h = h

        self.mu = mu             # mean molecular weight 
                                 # neutral hydrogen (76%) and helium (24%), mu = 50./41.
                                 # ionized hydrogen (76%) and helium (24%), mu = 10./17.
"bak/model_fbar_halo.py" 260L, 10356C                                                                                                  

