# Gog-Grenfell MultiStrain Model with Mutation, Coinfection and Waning
# Code developed by Gabriella Dantas Franco (2023)

import numpy as np
from operator import matmul
from scipy.integrate import odeint, simps


def PCI_Mut_coinf_waning(x, t, parms):
    '''
    Function that receives x=(S1, S2, ..., Snv, I1, I2, ..., R1, ..., Rnv)
    and gives us the dynamic equations SIR for variants with total cross immunity (TCI).
    parms must include 'nv' number of variants, 'mu' demography, 'beta'
    transmissibilty vector, 'gamma' recovery rate vector
    '''

    nv = parms['nv']
    S = x[ : nv]
    I = x[nv : 2*nv]
    #I[np.where(I < 1/parms['N'])] = 0
    R = x[2*nv : ]

    mu = parms['mu']
    omega = parms['omega']
    beta = parms['beta']
    gamma = parms['gamma']
    sigma = parms['sigmaprime']
    mutation = parms['mutation']

    dSdt = mu - np.sum(beta * I * S) - np.matmul(sigma, beta * I) * S + omega * R
    dIdt = (beta * S - gamma - mu) * I + np.matmul(mutation, I)
    dRdt = gamma * I - (mu + omega) * R + np.matmul(sigma, beta * I) * S
    dxdt = np.array([dSdt, dIdt, dRdt])

    return dxdt.flatten()