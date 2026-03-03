import os, sys
MAIN_DIR = "/Users/jacobgosselin/Library/CloudStorage/GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/research_ideas/negative_earnings"
os.chdir(MAIN_DIR)
sys.path.insert(0, os.path.join(MAIN_DIR, "code"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

from matplotlib import rc
rc('text', usetex=True)

import pdb # this is for debugging
#pdb.set_trace() # paste this in code for debugging

from time import time
from ProdFun_VPublic import ProdFun

#%% Options for estimation

## set which variables to include in the production function 
vars_how = OrderedDict() # ordered dictionary, coefficients ordered as the insertion order
vars_how['v'] = ['purge','static','variable'] # variable input, markup will be computed for this
vars_how['k'] = ['purge','dynamic','fixed'] # capital

# to add fixed costs add this line, and remove the line 70 that drops fixed costs
#vars_how['x'] = ['purge','dynamic','fixed'] # fixed cost

# to add variable p as a control in the first stage
#vars_how['p'] = ['purge','no input','no input'] # additional control in first stage 


#%% Import data

t0 = time()

# import data
dta_deep=pd.read_csv('data/intermediate/compustat_preMU.csv', parse_dates=[0])

# keep relevant variables
dta_deep = dta_deep[['firmid','date','sector','sale','varcost','capital','fixcost','v','k','y','x']]

# drop missing obs and fixed costs
dta_deep = dta_deep[(dta_deep.varcost > 0) & (dta_deep.capital > 0) & (dta_deep.sale > 0) & (dta_deep.fixcost > 0)]
dta_deep = dta_deep.drop(columns=['fixcost', 'x'])

# set date to integer, not datetime
# dta_deep['date'] = dta_deep['date'].dt.year

#%% Fit GMM and compute markups

t1 = time()

# creating the for loop: listing all sectors
ind_list = dta_deep.sector.unique()
ind_list = ind_list[np.logical_not(np.isnan(ind_list))]

# estimate the model
for ind in ind_list:
    
    print('')
    print('Industry ' + str(ind) + ' / ' + str(ind_list[-1]))
    
    # keep one industry
    dta_iter = dta_deep.loc[dta_deep['sector'] == ind]

    # select time window
    dta_iter = dta_iter[(dta_iter.date > 1959) & (dta_iter.date < 2024)]
    
    # estimate the model
    P = ProdFun(dta_iter, vars_how, translog=False)

    # store betas in a different dataframe
    #cols = []
    #cols.append('beta_c')
    cols = ['beta_c']
    for key in vars_how:
        if vars_how[key][1] != 'no input':
            #cols.append('beta_' + key)
            cols += ['beta_' + key]
                               
    beta_iter = pd.DataFrame([P.betas], columns = cols) 

    # add information to the stored betas
    beta_iter['sector'] = ind
    beta_iter['N_obs'] = dta_iter.shape[0]
    beta_iter['N_firms'] = dta_iter.firmid.unique().shape[0]
            
    # store AR(1) parameters for productivity
    ar1_iter = pd.DataFrame([{'sector': ind, 'rho': P.rho, 'sigma_xi': P.sigma_xi}])

    if ind == ind_list[0]: # first iteration of for loop
        dta_mu = P.dta_mu
        dta_betas = beta_iter
        dta_ar1 = ar1_iter
    else: # append for further iterations
        dta_mu = pd.concat([dta_mu, P.dta_mu], ignore_index=True)
        dta_betas = pd.concat([dta_betas, beta_iter], ignore_index=True)
        dta_ar1 = pd.concat([dta_ar1, ar1_iter], ignore_index=True)

dta_betas = dta_betas.sort_values(by=['sector'])

print('')
print('Resulting Betas:')    
print(dta_betas)    

t2 = time()

print('')
print('')
print('Elapsed time: ' + str(np.round(t2-t0,2)) + ' sec.')
print('Cleaning time: ' + str(np.round(t1-t0,2)) + ' sec.')
print('Fitting time: ' + str(np.round(t2-t1,2)) + ' sec.')
print('')
print('')
        

#%% create dataset to export

print('')
print('Exporting..')

# import data
dta_export=pd.read_csv('data/intermediate/compustat_preMU.csv', parse_dates=[0])
# set date to integer, not datetime
# dta_export['date'] = dta_export['date'].dt.year
                
# add markups
dta_merge01 = dta_mu[['firmid','date','mu_v']]
dta_export = dta_export.merge(dta_merge01
                            , on=['firmid','date'], how = 'left')      

dta_export = dta_export.sort_values(by=['sector','firmid','date']) # sorting
dta_check = dta_export.head(100) # to check the data
dta_export.to_csv('data/intermediate/compustat_postMU.csv', index=False)

# Consolidate AR(1) and production function parameters into ACF_bysector.csv
acf_bysector = dta_betas[['sector', 'beta_v', 'beta_k']].rename(
    columns={'sector': 'naics_2digit', 'beta_v': 'beta_l', 'beta_k': 'beta_k'}
)
acf_bysector = acf_bysector.merge(
    dta_ar1.rename(columns={'sector': 'naics_2digit'})[['naics_2digit', 'rho', 'sigma_xi']],
    on='naics_2digit'
)
acf_bysector = acf_bysector.sort_values(by='naics_2digit')
acf_bysector.to_csv('data/clean/ACF_bysector.csv', index=False)

print('')
print('Files exported')