###################################################################
# Copyright 2024 Haonan Zheng (nan.h.zheng@outlook.com)           #
# Our model is detailed in the appendix of our paper:             #
# https://arxiv.org/abs/2403.17044, or                            #
# https://ui.adsabs.harvard.edu/abs/2024arXiv240317044Z/abstract, #
# it would be great if you find it useful and cite it!            #
###################################################################

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

        self.H0 = 100. * self.h              # km/s/Mpc
        self.H0 = self.H0 * 1e3 / self.Mpc  # 1/s
        self.T0 = float(Ti) / np.power(1 + float(zi), 2)  # K

        # M_* = (0.255 * 5./3.)^-1.5 * (12 * pi)^-0.5 * cs0^3 * (G^-1.5 / Omega_m0 / rho_crit0) * rho_vir^0.5
        self.cs0 = np.sqrt(5./3. * self.kB * self.T0 / (self.mu * self.mp))  # m/s
        self.rho_crit0 = 3 * np.power(self.H0, 2) / (8 * np.pi * self.G)
        self.coeff = np.power(0.255 * 5./3., -1.5) * np.power(12. * np.pi, -0.5)
        if massdef == 'mean200':
            self.rho_vir = 200 * self.rho_crit0 * self.Omega_m * np.power(1 + z, 3)
        elif massdef == 'crit200':
            self.rho_vir = 200 * self.rho_crit0 * (self.Omega_m * np.power(1 + z, 3) + self.Omega_L)
        else:
            raise ValueError('define your own rho_vir...')
        self.ms = self.coeff * np.power(self.cs0, 3) * np.power(self.G, -1.5) * \
                  np.power(self.Omega_m * self.rho_crit0, -1) * np.power(self.rho_vir, 0.5) # unit: kg
        self.ms = self.ms / self.Msun
        print("m* = %1.3e M_sun" % self.ms)       
        
        # plot parameters
        self.color_list = ['#FF3030', '#FA9E5C', '#FCD12A', '#32CD32', \
                           '#40E0D0', '#1E90FF', '#DB98F1', '#552586']
        self.lw = 2

    def mt_fit(self, c): # mt: M_tilde = M_{1/2} / M_*
        # the fit is illustrated in self.plot_c_mt()
        a0 = -0.49101305
        a1 =  0.48944013
        a2 = -0.09876111
        a3 =  0.04950397
        a4 = -0.00589008

        log10c = np.log10(c)
        log10mt = a0 + a1 * log10c + a2 * np.power(log10c, 2) + \
                  a3 * np.power(log10c, 3) + a4 * np.power(log10c, 4)
        
        return np.power(10, log10mt)
    
    def mt_exact(self, c): # mt: M_tilde = M_{1/2} / M_*
        rct = fsolve(self.fbt_eq, np.power(c, 0.7), args=(c))[0]
        mt = np.power(np.power(rct, -1.28) * c * np.power(np.log(1 + c) - c / (1 + c), 1./3.), 1.5)
        return mt
    
    def mc_fit(self, c): # m_c: characteristic mass M_{1/2} at which f_b, halo / \bar{f}_b = 0.5
        return self.ms * self.mt_fit(c)
    
    def mc_exact(self, c): # m_c: characteristic mass M_{1/2} at which f_b, halo / \bar{f}_b = 0.5
        return self.ms * self.mt_exact(c)
    
    def m2fbt(self, m, c): # M_2 / M_*
        mt = m / self.ms
        rct = np.power(np.power(mt, 2./3.) / (c * np.power(np.log(1. + c) - c / (1. + c), 1./3.)), -1./1.28)
        return self.fbt(rct, c)
        
    ###### detailed calculation of f_b, halo / \bar{f}_b ######
    def ki(self, x):
        return self.gi(x) * np.power(x, -2) * (np.log(1. + x) - x / (1. + x))

    def gi(self, x):
        return np.power(x * (1. + x) * (1. + x), -1.)
    
    def hi(self, x):
        return np.log(1. + x) / x

    def k_exact(self, rt): # exact K(r_tilde)
        return quad(self.ki, rt, np.inf)[0]

    def k(self, rt): # approximate K(r_tilde)
        return self.l(rt) * np.power(rt, -5./3.) * np.power(1 + rt, -10./3.)
    
    def l_exact(self, rt):
        return np.power(rt, 5./3.) * np.power(1. + rt, 10./3.) * self.k_exact(rt)
    
    def l(self, rt):
        return 0.255 * np.power(rt, 1.28)

    def fi(self, rt, rct): # r_tilde, r_c_tilde

        gi_rct = self.gi(rct)
        gi_rt = self.gi(rt)

        hi_rct = self.hi(rct)
        hi_rt = self.hi(rt)

        k_rct = self.k(rct)

        return rt * rt * gi_rt * \
               (1 - (gi_rct / gi_rt) * \
                np.power(1 + 0.4 * (gi_rct / k_rct) * (hi_rt - hi_rct), 1.5))

    def f(self, rct, c):
        if not np.isscalar(rct):
            rct = rct[0]
        return quad(self.fi, 0, np.min([rct, c]), args=(rct))[0]

    def fbt(self, rct, c): # f_{b, halo} / \bar{f}_b
        return 1. - self.f(rct, c) / (np.log(1. + c) - c / (1. + c))

    def fbt_eq(self, rct, c):
        return self.fbt(rct, c) - 0.5
    
    def plot_mt_fbt(self, mta = np.logspace(-2, 2, 101), c_list=[64, 32, 16, 8, 4, 2]):
        fig = plt.figure(figsize=(6, 4))
        gs1 = gridspec.GridSpec(nrows=1, ncols=1, \
                                left=0.09, right=0.98, \
                                top=0.98, bottom=0.05, \
                                wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs1[0, 0])

        for c, color in zip(c_list, self.color_list):

            rcta = np.power(np.power(mta, 2./3.) / (c * np.power(np.log(1 + c) - c / (1 + c), 1./3.)), -1./1.28)

            fbta = [self.fbt(rct, c) for rct in rcta]
            ax1.plot(mta, fbta, label=r'$c = %d$' % c, color=color, lw=self.lw)

        ax1.legend(loc='upper left', frameon=False)
        ax1.set_xlabel(r'$M_\mathrm{vir}/M_*$')
        ax1.set_ylabel(r'$f_\mathrm{b, halo}/\bar{f}_\mathrm{b}$')
        ax1.set_xlim(np.power(10, -2.), np.power(10, 2.))
        ax1.set_ylim(0, 1.1)
        ax1.set_xscale('log')

        plt.savefig('mt-fbt.pdf', bbox_inches='tight')
        plt.show()
        return 

    def plot_rct_fbt(self, c_list=[64, 32, 16, 8, 4, 2]):
        fig = plt.figure(figsize=(6, 4))
        gs1 = gridspec.GridSpec(nrows=1, ncols=1, \
                                left=0.09, right=0.98, \
                                top=0.98, bottom=0.05, \
                                wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs1[0, 0])

        rcta = np.logspace(-1.5, 2.5, 101)
        for c, color in zip(c_list, self.color_list):
            fbta = [self.fbt(rct, c) for rct in rcta]
            ax1.plot(rcta, fbta, label=r'$c = %d$' % c, color=color, lw=self.lw)

        ax1.legend(loc='lower left', frameon=False)
        ax1.set_xlabel(r'$\tilde{r}_\mathrm{c}$')
        ax1.set_ylabel(r'$f_\mathrm{b, halo}/\bar{f}_\mathrm{b}$')
        ax1.set_xlim(np.power(10, -1.5), np.power(10, 2.5))
        ax1.set_ylim(0, 1.1)
        ax1.set_xscale('log')

        plt.savefig('rct-fbt.pdf', bbox_inches='tight')
        plt.show()
        return 
    
    def plot_c_mt(self):
        fig = plt.figure(figsize=(6, 4))
        gs1 = gridspec.GridSpec(nrows=1, ncols=1, \
                                left=0.09, right=0.98, \
                                top=0.98, bottom=0.05, \
                                wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs1[0, 0])

        ca = np.logspace(-1, 3.5, 251)
        rcta = []
        mta = []
        for c in ca:
            rct = fsolve(self.fbt_eq, np.power(c, 0.7), args=(c))[0]
            mt = np.power(np.power(rct, -1.28) * c * np.power(np.log(1 + c) - c / (1 + c), 1./3.), 1.5)
            rcta.append(rct)
            mta.append(mt)

        #print(np.transpose([ca, rcta, mta]))

        ax1.plot(ca, mta, color='black', lw=self.lw*1.25, \
                label=r'$M_\mathrm{vir}/M_*\ at\ f_\mathrm{b,\,halo}/\bar{f}_\mathrm{b}=0.5$')
        ax1.set_xlabel(r'$c$')
        ax1.set_ylabel(r'$M_\mathrm{vir}/M_*$')
        ax1.set_xlim(0.1, 100)
        ax1.set_ylim(0, 2.5)
        ax1.set_xscale('log')

        def mt_fit_func(log10c, a0, a1, a2, a3, a4):
            return a0 + a1 * log10c + a2 * np.power(log10c, 2) + \
                   a3 * np.power(log10c, 3) + a4 * np.power(log10c, 4)

        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(mt_fit_func, np.log10(ca), np.log10(mta), maxfev=100000)
        print('Numerical fit constant:', popt)
        ax1.plot(ca, np.power(10, mt_fit_func(np.log10(ca), *popt)), ls='--', c='w', lw=self.lw*2)
        ax1.plot(ca, np.power(10, mt_fit_func(np.log10(ca), *popt)), ls='--', c='darkgray', lw=self.lw, \
                 label='$numerical\ fit$')

        ax1.legend(loc='upper left', frameon=False)

        plt.savefig('c-mt.pdf', bbox_inches='tight')
        plt.show()
        return 


model = model_fbar_NFW(z = 3.07, massdef='mean200')
#model = model_fbar_NFW(z = 1.97, massdef='mean200', Omega_m=0.3, h=0.7, Ti=1e7, mu=10./17.)

c = 10.0

# example 1 (calculate M_{1/2}, z = 3.07, c = 10)
print('M_{1/2}=%1.3e M_sun for c = %1.1f' % (model.mc_fit(c = c), c))

# example 2 (calculate f_b, halo / \bar{f}_b, z = 3.07, c = 10)
m = model.mc_fit(c = c)
print('f_b, halo / \\bar{f}_b = %1.3f for M = %1.3e M_sun, c = %1.1f' % (model.m2fbt(m = m, c = c), m, c))
m = model.mc_exact(c = c)
print('f_b, halo / \\bar{f}_b = %1.3f for M = %1.3e M_sun, c = %1.1f' % (model.m2fbt(m = m, c = c), m, c))

# plot
print('Plotting M_{vir}/M_* vs. f_b, halo / \\bar{f}_b...')
model.plot_mt_fbt()

print('Plotting r_c_tilde vs. f_b, halo / \\bar{f}_b...')
model.plot_rct_fbt()

print('Plotting c vs. M_{1/2}/M_*...')
model.plot_c_mt()
