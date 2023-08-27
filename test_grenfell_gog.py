# Code developed by Gabriella Dantas Franco and Denise Cammarota
# Reproducing the Grenfell and Gog values plots

import numpy as np
from operator import matmul
from scipy.integrate import odeint, simps
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.stats as st
from scipy.stats import entropy


from Dyn_strains_M_CI_mut import Dyn_strains_M_CI_mut

# Parameters for simulation
pars = {}
pars['nv'] = 100     # number of variants 
mu =  (1.0)                                   
pars['mu'] = (mu) * np.ones(pars['nv'])                     # demography
pars['gamma']= np.linspace(0.5, 0.5, pars['nv'])                # recovery rate (1/days)
pars['beta'] = np.linspace(3.0, 3.0, pars['nv'])*pars['gamma'] # feasible betas
pars['basR0'] = pars['beta']/(pars['gamma']+ mu)                   # R0 number 

# Cross-immunity matrix calculation
pars['sigma'] = np.zeros((pars['nv'], pars['nv']))
pars['mutation'] = np.zeros((pars['nv'], pars['nv']))
m = (0.02) # m mutation rate
d = 10 # d typical distance between variants for cross immunity 

for i in range(pars['nv']):
    pars['mutation'][i,i] = -2*m
    for j in range(pars['nv']):
        pars['sigma'][i,j] = np.exp(-((i-j)/d)**2)
        if np.abs(i-j)==1:
            pars['mutation'][i,j] = m
            
pars['N'] = 1000000
# população de infectado não nula apenas para a variante 1 (=10)
pars['I0'] = np.zeros(pars['nv'])
pars['I0'][0] = 10000
pars['S0']  =  pars['N'] * np.ones(pars['nv']) - np.sum(pars['I0'])

# Definindo o vetor de condições iniciais com unidimensional da forma (S1, I1, R1, S2, I2, R2, ... , Snv, Inv, Rnv)
x0 = np.zeros(2*pars['nv'])
x0[: pars['nv']] = pars['S0']/pars['N']
x0[pars['nv'] : 2*pars['nv']] = pars['I0']/pars['N']


# Solving with different hypotesis

## With coinfection (original models)
t = np.arange(0.0, 10000, 0.1)
sol1 = odeint(Dyn_strains_M_CI_mut, x0, t, args =(pars,))

### Visualization 1
plt.figure(figsize=(10,8))
for i in range(pars['nv']):
    #plt.plot(t, sol[:, i], label = '$S_{%i}$' % i)
    #plt.plot(t, sol1[:, i], label = '$S_{%i}$' % i)
    plt.plot(t, sol1[:, pars['nv']+i], label = '$I_{%i}$' % (i+1), linewidth = 3)
#plt.legend(loc='best', fontsize = 14)
plt.xlabel('Time (days)', fontsize = 18)
plt.ylabel(r'$I_i$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.ylim((-0.001,0.05))
plt.savefig('figs/test_EpiCurves.pdf')
plt.show()


### Visualization 2

####  Proportion of infecteds
#plt.figure(figsize = (10,8))
St = np.zeros((pars['nv'], len(t)))
It = np.zeros((pars['nv'], len(t)))
for j in range(pars['nv']):
    St[j] = sol1[:, j]
    It[j] = sol1[:, pars['nv']+j]
    
cmap = 'inferno'
for i in range(pars['nv']):
    plt.fill_between(t, 0, np.sum(It[i:], axis=0)/np.sum(It, axis=0),
                     color = sns.color_palette(cmap, pars['nv'])[i],
                    label = f'Strain %i' % i, edgecolor = 'k', alpha = 0.5)

plt.xlabel('Time (days)', fontsize = 14)
plt.ylabel(r'$\frac{I_i}{I_T}$', fontsize = 14, rotation = 0, labelpad=16)
plt.savefig('figs/test_FreqCurves.pdf')
plt.show()
#plt.legend()

### Visualization 3: Number of strains
plt.subplots(figsize = (10,8))
It_bin = (It > 10**-6)
It_sum = It_bin.sum(axis = 0)
plt.plot(t,It_sum, linewidth = 4)
plt.xlabel('Time (days)', fontsize = 18)
plt.ylabel(r'$n_t$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.savefig('figs/test_NumberVariants.pdf')
plt.show()

### Visualization 4: R0 of the ones on the population for each time 
### Or at least mean R0
R0_t = []
for i in range(len(t)):
    R0_t.append(pars['basR0'][It_bin[:,i]])

### calculate max, min and average over time
max_t = []
min_t = []
avg_t = []
for i in range(len(t)):
    if(len(R0_t[i]) != 0):
        max_t.append(max(R0_t[i]))
        min_t.append(min(R0_t[i]))
        avg_t.append((R0_t[i]).mean())
    else:
        max_t.append(np.nan)
        min_t.append(np.nan)
        avg_t.append(np.nan)

plt.figure(figsize = (10,8))
plt.plot(t,max_t, linewidth = 4, label = 'Maximum '+r'$R_0$')
plt.plot(t,min_t, linewidth = 4, label = 'Minimum '+r'$R_0$')
plt.plot(t,avg_t, linewidth = 4, label = 'Mean '+r'$R_0$')
plt.ylim((1.5,4.1))
plt.xlabel('Time (days)', fontsize = 18)
plt.ylabel(r'$R_0$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.savefig('figs/test_ValuesR0.pdf')
plt.show()

