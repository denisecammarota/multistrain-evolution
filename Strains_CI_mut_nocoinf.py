# Gog-Grenfell MultiStrain Model with Mutation and No Coinfection
# Code developed by Gabriella Dantas Franco (2023)

import numpy as np
from operator import matmul
from scipy.integrate import odeint, simps

def Strains_CI_mut_nocoinf(x, t, parms):
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

    mu = parms['mu']
    beta = parms['beta']
    gamma = parms['gamma']
    sigma = parms['sigmaprime']
    mutation = parms['mutation']
    dSdt = mu - np.sum(beta * I * S) - np.matmul(sigma, beta * I) * S - mu * S + np.sum(gamma * I) - gamma * I
    dIdt = (beta * S - gamma - mu) * I + matmul(mutation, I)
    dxdt = np.array([dSdt, dIdt])

    return dxdt.flatten()