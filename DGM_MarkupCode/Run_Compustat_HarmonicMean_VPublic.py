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

#%% 
############################
# Options for estimation
############################

## set which cariables to include in the production function 
vars_how = OrderedDict() # ordered dictionary, coefficients ordered as the insertion order
vars_how['v'] = ['purge','static','variable'] # variable input, markup will be computed for this
vars_how['k'] = ['purge','dynamic','fixed'] # capital

# to add fixed costs add this line, and remove the line 70 that drops fixed costs
#vars_how['x'] = ['purge','dynamic','fixed'] # fixed cost

# to add variable p as a control in the first stage
#vars_how['p'] = ['purge','no input','no input'] # additional control in first stage 




#%% 
############################
# Import data
############################
t0 = time()

# import data
dta_deep=pd.read_csv('COMPUSTAT_Clean.csv', parse_dates=[0])

# keep relevant variables
dta_deep = dta_deep[['firmid','date','sector','sale','varcost','capital','fixcost','v','k','y','x']]

# drop missing obs and fixed costs
dta_deep = dta_deep[(dta_deep.varcost > 0) & (dta_deep.capital > 0) & (dta_deep.sale > 0) & (dta_deep.fixcost > 0)]
dta_deep = dta_deep.drop(columns=['fixcost', 'x'])

#%%
#############################
# Fit GMM and compute markups
#############################
t1 = time()

# creating the for loop: listing all sectors
ind_list = dta_deep.sector.unique()
ind_list = ind_list[np.logical_not(np.isnan(ind_list))]
## just for one sector
#ind_list=[33] #force to estimate on 1 sector: Manufacturing transportation
#ind_list=[22] #force to estimate on 1 sector: Utilities
## select a list of sectors
#ind_list = ind_list[0:4]

# estimate the model
for ind in ind_list:
    
    print('')
    print('Industry ' + str(ind) + ' / ' + str(ind_list[-1]))
    
    # keep one industry
    dta_iter = dta_deep.loc[dta_deep['sector'] == ind]
    
    # select time window
    dta_iter = dta_iter[(dta_iter.date > 1959) & (dta_iter.date < 2017)]
    
    # estimate the model
    P = ProdFun(dta_iter, vars_how 
                )

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
            
    if ind == ind_list[0]: # first iteration of for loop
        dta_mu = P.dta_mu
        dta_betas = beta_iter
    else: # append for further iterations
        #dta_mu = dta_mu.append(P.dta_mu)
        #dta_betas = dta_betas.append(beta_iter)
        dta_mu = pd.concat([dta_mu, P.dta_mu], ignore_index=True)
        dta_betas = pd.concat([dta_betas, beta_iter], ignore_index=True)

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

#%% 
#################################
## Rolling Window GMM and markups
#################################

print('')
print('Estimating 7 year rolling window')

t1 = time()

# creating the for loop: listing all sectors
ind_list = dta_deep.sector.unique()
ind_list = ind_list[np.logical_not(np.isnan(ind_list))]

# estimate the model
for ind in ind_list:
    
    print('')
    print('Industry ' + str(ind) + ' / ' + str(ind_list[-1]))
    
#    pdb.set_trace() # paste this in code for debugging

    # keep one industry
    dta_iter_0 = dta_deep.loc[dta_deep['sector'] == ind]
    
    year_counter = 0
    for year in range( max(1960, min(dta_iter_0.date.unique())+2 ) , 1 + min(2016, max(dta_iter_0.date.unique())-2 ) ):
        year_counter += 1
        
        # select time window
        dta_iter = dta_iter_0[(dta_iter_0.date > year-4) & (dta_iter_0.date < year+4)]
            
        # estimate the model
        P = ProdFun(dta_iter, vars_how
                    ,verbose=False
                    )

        # store betas in a different dataframe
        cols = []
        cols.append('beta_c')
        for key in vars_how:
            if vars_how[key][1] != 'no input':
                cols.append('beta_' + key)
    
     
        beta_iter = pd.DataFrame([P.betas], columns = cols) 

        # add information to the stored betas
        beta_iter['sector'] = ind
        beta_iter['year'] = year
            
        if ind == ind_list[0] and year_counter==1: # first iteration of for loop
            dta_mu_roll = P.dta_mu[P.dta_mu.date == year]
            dta_betas_roll = beta_iter
        else: # append for further iterations
            #dta_mu_roll = dta_mu_roll.append(P.dta_mu[P.dta_mu.date == year])
            #dta_betas_roll = dta_betas_roll.append(beta_iter)    
            dta_mu_roll = pd.concat([dta_mu_roll, P.dta_mu[P.dta_mu.date == year]], ignore_index=True)
            dta_betas_roll = pd.concat([dta_betas_roll, beta_iter], ignore_index=True)
    

dta_betas_roll = dta_betas_roll.sort_values(by=['sector','year'])

t2 = time()

print('')
print('')
print('Fitting time: ' + str(np.round(t2-t1,2)) + ' sec.')
print('')
print('')

#%% Plotting Average Markup

# set trimming percentage
trim = 0.04

# plot one time series of markup for each variable input
for key in vars_how:
    if vars_how[key][2] == 'variable':
        interim_mu = 'mu_' + key
        
        # trim the dataset on costshare (Rolling windows)
        dta_mu_trimmed_roll = dta_mu_roll.copy()
        #dta_mu_trimmed = dta_mu.copy()
        dta_mu_trimmed_roll['costshare'] = dta_mu_trimmed_roll['varcost'] / (dta_mu_trimmed_roll['varcost'] + dta_mu_trimmed_roll['capital'])
        dta_mu_trimmed_roll['pc_low'] = dta_mu_trimmed_roll[['costshare','date']].groupby(['date']).transform(lambda x: x.quantile(trim/2))
        dta_mu_trimmed_roll['pc_high'] = dta_mu_trimmed_roll[['costshare','date']].groupby(['date']).transform(lambda x: x.quantile(1-trim/2))
        dta_mu_trimmed_roll = dta_mu_trimmed_roll[(dta_mu_trimmed_roll['costshare'] > dta_mu_trimmed_roll['pc_low']) & (dta_mu_trimmed_roll['costshare'] < dta_mu_trimmed_roll['pc_high'])]
        
        # trim the dataset on costshare (Non Rolling windows)
        dta_mu_trimmed = dta_mu.copy()
        dta_mu_trimmed['costshare'] = dta_mu_trimmed['varcost'] / (dta_mu_trimmed['varcost'] + dta_mu_trimmed['capital'])
        dta_mu_trimmed['pc_low'] = dta_mu_trimmed[['costshare','date']].groupby(['date']).transform(lambda x: x.quantile(trim/2))
        dta_mu_trimmed['pc_high'] = dta_mu_trimmed[['costshare','date']].groupby(['date']).transform(lambda x: x.quantile(1-trim/2))
        dta_mu_trimmed = dta_mu_trimmed[(dta_mu_trimmed['costshare'] > dta_mu_trimmed['pc_low']) & (dta_mu_trimmed['costshare'] < dta_mu_trimmed['pc_high'])]
        
        # some stats on markups
        print('Markups for variable input ' + key)
        print('After Trimming ' + str(trim*100) + '% of observations')
        print('Average Markup = ' + str(round(dta_mu_trimmed[interim_mu].mean(),4)) +
              ' ; Median Markup = ' + str(round(dta_mu_trimmed[interim_mu].median(),4)))
        
        dta_mu_wts = pd.DataFrame(dta_mu_trimmed[[interim_mu,'date','sale']].groupby('date').apply(lambda x: np.average(x[interim_mu], weights=x.sale)))
        dta_mu_wts.columns = ['mu_wts']
        dta_mu_wts['date']=dta_mu_wts.index
        
        
        dta_mu_roll_wts = pd.DataFrame(dta_mu_trimmed_roll[[interim_mu,'date','sale']].groupby('date').apply(lambda x: np.average(x[interim_mu], weights=x.sale)))
        dta_mu_roll_wts.columns = ['mu_wts']
        dta_mu_roll_wts['date']=dta_mu_roll_wts.index
                
        # define arguments to format into the figures
        args = {
                'arg0': '{Mark-up}',
                'arg1': key
                }
        # plot sales weighted Markup
        plt.plot(dta_mu_wts['date'], dta_mu_wts['mu_wts'], color = 'blue') 
        plt.plot(dta_mu_wts['date'], dta_mu_roll_wts['mu_wts'], color = 'red') 
        plt.axhline(y = 1.0, linestyle = '--', color = 'black')
#        plt.title(r'\textsc{arg0}: Sales-Weighted Average for ${arg1}$'.format(**args)) 
#        plt.xlabel(r'\textsc{Year}')
#        plt.legend([r'Constant Production Function',r'5 Year Rolling Windows'])
        plt.title('{arg0}: Sales-Weighted Average for {arg1}'.format(**args)) 
        plt.xlabel('Year')
        plt.legend(['Constant Production Function','5 Year Rolling Windows'])
        plt.grid(True)
        plt.show()
        
        # Compute harmonic mean for non-rolling data
        dta_mu_harmonic = pd.DataFrame(dta_mu_trimmed.groupby('date').apply(
            lambda x: x['sale'].sum() / (x['sale'] / x[interim_mu]).sum()))
        dta_mu_harmonic.columns = ['mu_harmonic']
        dta_mu_harmonic['date'] = dta_mu_harmonic.index

        # Compute harmonic mean for rolling data
        dta_mu_roll_harmonic = pd.DataFrame(dta_mu_trimmed_roll.groupby('date').apply(
            lambda x: x['sale'].sum() / (x['sale'] / x[interim_mu]).sum()))
        dta_mu_roll_harmonic.columns = ['mu_harmonic']
        dta_mu_roll_harmonic['date'] = dta_mu_roll_harmonic.index
                
        # Define arguments to format into the figures
        args = {
            'arg0': '{Mark-up}',
            'arg1': key
        }
        
        # Plot sales-weighted harmonic mean of markup
        plt.plot(dta_mu_harmonic['date'], dta_mu_harmonic['mu_harmonic'], color='blue')
        plt.plot(dta_mu_roll_harmonic['date'], dta_mu_roll_harmonic['mu_harmonic'], color='red')
        plt.axhline(y=1.0, linestyle='--', color='black')
        plt.title('{arg0}: Sales-Weighted Harmonic Mean for {arg1}'.format(**args)) 
        plt.xlabel('Year')
        plt.legend(['Constant Production Function', '5 Year Rolling Windows'])
        plt.grid(True)
        plt.show()
        
        
        

#%% create dataset to export

print('')
print('Exporting..')

# import data
dta_export=pd.read_csv('COMPUSTAT_Clean.csv', parse_dates=[0])
                
# add markups
dta_merge01 = dta_mu[['firmid','date','mu_v']]
dta_export = dta_export.merge(dta_merge01
                            , on=['firmid','date'], how = 'left')      

dta_merge02 = dta_mu_roll[['firmid','date','mu_v']]
dta_export = dta_export.merge(dta_merge02
                            , on=['firmid','date'], how = 'left')      

dta_export = dta_export.sort_values(by=['sector','firmid','date']) # sorting

dta_check = dta_export.head(100) # to check the data

dta_export.to_stata('Compustat_Markups_Python_V1.dta', write_index=False)
dta_export.to_csv('Compustat_Markups_Python_V1.csv', index=False)

# export coefficients of the PF
dta_betas.to_stata('Compustat_Betas_Python_V1.dta', write_index=False)
dta_betas_roll.to_stata('Compustat_BetasRoll_Python_V1.dta', write_index=False)

# export time series for aggregate markup
dta_mu_wts.to_csv('Average_Markup.csv', index=False)
dta_mu_roll_wts.to_csv('Average_Markup_Roll.csv', index=False)

# export time series for harmonic mean markup
dta_mu_harmonic.to_csv('Harmonic_Markup.csv', index=False)
dta_mu_roll_harmonic.to_csv('Harmonic_Markup_Roll.csv', index=False)

print('')
print('Files exported')        