"""
This class estimates production function parameters and firm level markups.

@author: Giovanni Morzenti (giomorzent@live.it)
@author: Basile Grassi

"""
############################
# Import modules & Packages
############################
import numpy as np
import pandas as pd
import warnings
#from scipy import stats 
from scipy import optimize
from time  import time
#import statsmodels.api as sm
############################
# Define the class
############################
class ProdFun:
    """
    This class estimates a Production Function
    using the procedure by Ackerberg, Caves & Frazer (2007,2015)
    which is a variation over Olley & Pakes (1996) and Levinsohn
    & Petrin (2003).
    
    The procedure is in 2 stages:
        1) Estimate non-parametrically output against a set of variables. These
           always include the production inputs, and possibly some other controls;
        2) Construct a GMM objective function w/n-moments, n being the number of 
           production inputs, and minimize it.
    
    Ultimately, the class estimates mark-ups (for variable inputs only)
    à la De Ridder, Grassi and Morzenti (2022), who build on De Loecker & Warzinsky (2012).
    
    Input
    -----------------
    data: pandas.dataframe ['firmid', 'date', 'varcost', 'capital', 'sale', 'sector_2d', 'v', 'k', 'y']
        ['firmid']: identifier of firms (GVKEY in Compustat)
        ['date']: date of each observation, can be different from yearly
        ['varcost']: deflated variable input (COGS in Compustat) 
        ['capital']: deflated capital (PPENT in Compustat) 
        ['sale']: deflated sales (SALE in Compustat)
        ['sector']: industry
        ['v']: log of 'varcost'
        ['k']: log of 'capital'
        ['y']: log of 'sale'
        ['x']: log of variable (optional, could be any other variable)
            
    vars_how: dict, optional (default = {'k':['purge','dynamic','fixed'], 'v':['purge','static','variable']})
        Select inputs and provide options for thier use in the estimation procedure    
        
        {       'name'        : [   'purge'/'no purge'    ,      'no input'/'static'/'dynamic'      ,     'no input'/'variable'/'fixed']}
            name of variable         include in purge        input: not included,  static, dynamic            markup: no/yes/no
        
        ADVANCED: if you wish to select the order of the inputs you can use an ordered dictionary (see OrderedDict())
    
    M: int, optional (default = 3)
        order of interaction terms for inputs used in purging regression
        
    purging: bool, optional (default = True)
        True: purge sales by a regression with multiple inputs interactions
        False: no purging, and consequently no sales correction
        
    phi_true: string, optional (default = '')
        '': compute \phi in the first stage using the standard procedure
        'var': use self.data['var'] as true value of \phi, instead of computing it withthe first stage
        NOTE: it works only if purging = False
        
    FE: list(str), optional (default = [])
        Select which variable to use as FE in purging and initial values of GMM
        []: No fixed effects
        ['date']: year fixed effects
        ['firmid_FE']: firm fixed effects (noticeably slower)
        ['date','firmid_FE']: firm and date fixed effects (noticeably slower)
        
    FE_demean: boolean, optional (default = True)
        True: use demeaning for Fixed Effects
        False: use dummy variables for Fixed Effects
        
    init_FE: boolean, optional (default = False)
        True: add selected FE to the OLS eqaution of the first stage (purge)

    translog: bool, optional (default = True)
        True: adding interactions terms (up to second power, e.g. v^2) 
              allowing for non-linearity in the production function
    
    tl_inter: bool, optional (default = False)
        True: add interaction terms of power 1 between different inputs (e.g. v*k)
        
    GMM_cons: bool, optional (default = True)
        True: include the constant in the GMM estimation
        
    beta_init: np array, optional (default = [])
        []: compute initial values with OLS
        Otherwise, this allows to specify the intial values of the GMM estimation. For example:
        np.array([1, 0.4, 0.6])
                        
    init_control: list of strings, optional (default = [])
        include additional controls in the OLS regression to compute initial values    
        []: include no additional regressior
        ['p','ms']: include variables 'p' and 'ms' as additional regressors
        
    optim: str, optional (default = 'NM')
        Select which optimizer to use
        'NM': for the Nelder-Mead simplex algorithm
        'BFGS': for quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno
        'FSOLVE': for MINPACK’s hybrd and hybrj algorithms to find roots of non linear system of equations 
        'BASINHOPPING': for the basin-hopping algorithm (global minimum, slow)
        
        ADVANCED: Try several optimizers if 'NM' does not converge
        'Iter': Executes 'NM',
            if the optimization is not successfull executes 'FSOLVE',
            if the optimization is not successfull executes 'BASINHOPPING'
    
    AR_c: bool, optional (default = True)
        True: Includes the constant in the AR used to compute moments for GMM
        
    AR_sqlag: bool, optional (default = False)
        True: Includes squared lag in the AR used to compute moments for GMM
                
    demean: bool, optional (default = False)
        True: demeans variables before first stage, initial values and GMM
        
    NormMoments: bool, optional (default = False)
        True: Renormalize moments so as to have correlation rather than covariance

    LogMoments: bool, optional (default = False)
        True: Take log of moments before giving them to the minimizer
        NOTE: This is not implemented for FSOLVE (it is not a minimizer, but an equation solver)
                                
    markups: bool, optional (default = True)
        True: computes firm specific markups and stores them in self.dta_mu
        False: does not compute markups (recommended for low memory usage)
    
    sales_correction: bool, optional (default = False)
        True: applies sales correction in the computation of markups
    
    verbose: bool, optional (default = True)
        True: Displays several metrics and results of the fitting
        False: Displays nothing, and stores all results in the object        
                                    
    
    Output
    -----------------
    self.dta: pandas.dataframe
        Dataframe used in the estimation, containing all interaction and lagged variables
        in this dataframe the first year is dropped for each firm due to lagged variables
                
    self.dta_mu: pandas.dataframe
        ['date','sale','varcost','sector_2d','epsilon','mu']
        Dataframe containing markups ['mu'], computed per each firm-date observation
        Markups are computed also for the years dropped in self.dta
    
    self.b_ols: np.array
        Initial values for elasticities estimated with GMM
    
    self.betas: np.array
        Elasticities estimated with GMM
        
    self.conv: tuple
        Full output of the optimizer (contains useful informations on convergence)
            
    """
    
    def __init__(self
                 ,data
                 ,vars_how = {'v':['purge','static','variable'], 'k':['purge','dynamic','fixed']}
                 ,M=3
                 ,purging=True
                 ,phi_true=''
                 ,FE=[]
                 ,FE_demean = True
                 ,init_FE = True
                 ,translog = False
                 ,tl_inter = False
                 ,GMM_cons = True
                 ,beta_init = []
                 ,init_control = []
                 ,optim='NM'
                 ,AR_c=True
                 ,AR_sqlag=False
                 ,demean=False
                 ,NormMoments=False
                 ,LogMoments=False
                 ,markups=True
                 ,sales_correction=False
                 ,verbose=True
                 ):
        
        # check for errors in vars_how        
        for key in vars_how:
            if len(vars_how[key]) != 3:
                raise KeyError('Please specify all necessary information for {} (see help).'.format(key))
            
            for element in vars_how[key]:
                if element not in ['no input', 'purge', 'no purge', 'variable', 'fixed', 'static', 'dynamic']:
                    raise KeyError('Keyword in {} is not valid (see help)'.format(key))
                    
            if vars_how[key][0] == 'no purge':
                
                if vars_how[key][1] != 'no input':
                    raise ValueError('{} is not included in the first stage so it cannot be an input.'.format(key))
                    
                elif vars_how[key][2] != 'no input':
                    raise ValueError('{} is not included in the first stage so it can be neither a variable nor a fixed input.'.format(key))
                    
            elif vars_how[key][0] == 'purge' and vars_how[key][1] != 'no input':
                
                if vars_how[key][2] == 'no input':
                    raise ValueError('{} is an input, specify whether it is fixed or variable.'.format(key))

        """
        Functions are executed here.
        Workflow below.
        """       

        # store parameters
        self.data = data
        self.vars_how = vars_how
        self.M = M
        self.purging = purging
        self.phi_true = phi_true
        self.FE = FE
        self.FE_demean = FE_demean
        self.init_FE = init_FE
        self.translog = translog
        self.tl_inter = tl_inter
        self.GMM_cons = GMM_cons
        self.beta_init = beta_init
        self.init_control = init_control
        self.optim = optim
        self.AR_c = AR_c
        self.AR_sqlag = AR_sqlag
        self.demean = demean
        self.NormMoments = NormMoments
        self.LogMoments = LogMoments
        self.markups = markups
        self.sales_correction = sales_correction
        self.verbose = verbose


        # load the data
        self.dta = data
        # Initial cleaning and purging of sales
        self.purge()
        # initial values
        self.initial_values()
        # initialize GMM
        self.initialize_GMM()
        
        # fit GMM
        self.fit_GMM()
        # recover AR(1) parameters for productivity
        self.compute_ar1()

        if self.markups:
            # compute markups
            self.compute_markups()        
            # clean the dataset for output
            self.clean_dta_mu()
            
        
    #----------------------------------------------------------#  
    def purge(self):
        """
        This is the first stage in ACF(2007), namely to regress
        firm level output against a set of variables included
        in the purge.
        Regression here is non-parametric using a M-order
        polynomial. 
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dta['firmid_FE'] = self.dta['firmid']
            self.dta['date_FE'] = self.dta['date']
        
        #set up the panel data structure
        self.dta = self.dta.set_index(['firmid','date'], drop=False)
        self.dta = self.dta.drop(columns=['firmid'])
        
        # drop missing obs
        self.dta = self.dta.replace([np.inf, -np.inf], np.nan).dropna()
        
        # store dataset for computation of firm level markups        
        if self.markups:
            self.dta_mu = self.dta
                    
        # create the constant for OLS estimation         
        self.dta['const'] = 1 # add constant
    
        if self.purging == True:            

            # select which variables to include in purging stage
            vars_purging = []
            for key in self.vars_how:
                if self.vars_how[key][0] == 'purge' and self.vars_how[key][1] != 'no input':
                    vars_purging.append(key)
                    
            # create interactions
            self.interlist = []
            
            if self.GMM_cons:
                self.interlist.append('const')
            
            # additional controls
            for key in self.vars_how:
                if self.vars_how[key][0] == 'purge' and self.vars_how[key][1] == 'no input':
                    self.interlist.append(key) # just the control

            # interaction terms
            vars_purging2=vars_purging.copy()
            for var in vars_purging:
                for i in range(1,self.M+1):
                    self.dta[var + str(i)] = self.dta[var].pow(i)
                    self.interlist.append(var + str(i))
                    vars_purging3=vars_purging2.copy()
                    for var2 in vars_purging2:
                         if var2 != var:
                             for j in range(1,self.M+1):
                                 if i+j<=self.M:
                                     self.dta[var + str(i) + var2 + str(j)] = self.dta[var].pow(i)*self.dta[var2].pow(j)
                                     self.interlist.append(var + str(i) + var2 + str(j))
                                 for var3 in vars_purging3:
                                     if var3 != var and var3 != var2:
                                         for k in range(1,self.M+1):
                                             if i+j+k <= self.M:
                                                 self.dta[var + str(i) + var2 + str(j)+ var3 + str(k)] = self.dta[var].pow(i)*self.dta[var2].pow(j)*self.dta[var3].pow(k)
                                                 self.interlist.append(var + str(i) + var2 + str(j)+ var3 + str(k))
                         vars_purging3.remove(var2)
                vars_purging2.remove(var)
            
            # Prepare matrices for OLS regression
            Y = self.dta[['y']]
            X = self.dta[self.interlist+self.FE]
            
            if self.FE_demean: # use demeaning for FE
                X = X.drop(columns=self.FE)
                X_pred = np.array(X) # store it to compute phi and epsilon
                Y_pred = np.array(Y) # store it to compute phi and epsilon
                if self.FE != []: # demean for desired fixed effects
                    if len(self.FE)<3:
                        for var_FE in self.FE:
                            X = X - X.replace([np.inf, -np.inf], np.nan).dropna().mean(level=var_FE) # demean the variable
                            Y = Y - Y.replace([np.inf, -np.inf], np.nan).dropna().mean(level=var_FE) # demean the variable
                    else: print('Impossible to demean with more than 2 FE levels')
            
            else: # use dummy variables for FE
                if self.FE != []: # add desired fixed effects
                    if self.GMM_cons:
                        dummies = pd.get_dummies( sum(X[self.FE].values.tolist(),[]) , drop_first=True )
                    else:
                        dummies = pd.get_dummies( sum(X[self.FE].values.tolist(),[]) , drop_first=False )
                    X = pd.concat([X.reset_index(drop=True),dummies.reset_index(drop=True)], axis=1)
                    Y = Y.reset_index(drop=True)
                    X = X.drop(columns=self.FE)
                X_pred = np.array(X) # store it to compute phi and epsilon
                Y_pred = np.array(Y) # store it to compute phi and epsilon

            C=pd.concat([X,Y],axis=1).replace([np.inf, -np.inf], np.nan).dropna() # drop missing values
            X = C.drop(columns=['y']); Y = C[['y']] # get back X and Y
                
            X = np.array(X)
            Y = np.array(Y) # turn Y into an array as well
            
            if self.demean:
                Y = Y - np.nanmean(Y)
                if self.FE != [] and self.FE_demean==False:
                    if self.GMM_cons:
                        X[:,1:len(self.interlist)] = X[:,1:len(self.interlist)] - np.nanmean(X[:,1:len(self.interlist)], axis=0)
                    else:
                        X[:,0:len(self.interlist)] = X[:,0:len(self.interlist)] - np.nanmean(X[:,0:len(self.interlist)], axis=0)
                elif self.FE_demean:
                    pass
                else:
                    X = X - np.nanmean(X, axis=0)
            
            
            #Perform OLS regression
            betas = np.linalg.pinv(X) @ Y
            predict = X_pred @ betas
            
            #residual
            epsilon = Y_pred - predict

            self.Y_purge = Y_pred
            self.Y_predict = predict

            self.Rsq_purge = 1 - ( np.var(epsilon[np.logical_not(np.isnan(epsilon))]) / np.var(Y_pred[np.logical_not(np.isnan(Y_pred))]) )
            if self.verbose:
                print('First Stage R squared = ' + str(self.Rsq_purge.round(4)))
            
            self.dta['phi'] = predict
            self.dta['epsilon'] = epsilon
        
        else:
            self.dta['phi'] = self.dta['y'] 
            self.dta['epsilon'] = 0
            
            if self.phi_true!='':
                print('No first stage, phi substituted with ' + self.phi_true)
                self.dta['phi'] = self.dta[self.phi_true] 
                self.dta['epsilon'] = self.dta['y'] - self.dta[self.phi_true]
                
    
   #----------------------------------------------------------# 
    def initial_values(self):
        """
        Construct the initial conditions to give to the GMM optimizer.
        """
        
        #sort data by firmsId and year
        self.dta=self.dta.sort_values(by=['firmid_FE','date_FE'])

        # constuct lags of purged sales
        self.dta['phi_lag'] = self.dta.groupby(['firmid'])['phi'].shift(1)
        
        # constuct lags of inputs
        for key in self.vars_how:
            if self.vars_how[key][1] != 'no input':
                self.dta[key + '_lag'] = self.dta.groupby(['firmid'])[key].shift(1)
        
        if self.translog == True:
            
            var_list_support = [] # list needed for interaction terms
            for key in self.vars_how:
                if self.vars_how[key][1] != 'no input':
                    var_list_support.append(key)
                    
            for key in self.vars_how:
                if self.vars_how[key][1] != 'no input':
                    
                    # construct square terms
                    self.dta[key + '2'] = self.dta[key].pow(2)
                    self.dta[key + '_lag2'] = self.dta[key + '_lag'].pow(2)
                    
                    if self.tl_inter: # construct interaction terms
                        for key2 in var_list_support: 
                            if key2 != key:
                                self.dta[key + key2] = self.dta[key] * self.dta[key2]
                                self.dta[key + '_lag' + key2] = self.dta[key+'_lag'] * self.dta[key2]
                                self.dta[key + key2 + '_lag'] = self.dta[key] * self.dta[key2+'_lag']
                                self.dta[key + '_lag' + key2 + '_lag'] = self.dta[key+'_lag'] * self.dta[key2+'_lag']
                        
                        var_list_support.remove(key) # drop variable from list, so as not to duplicate terms
                            
        # drop missing obs (one obs per firm)
        self.dta = self.dta.replace([np.inf, -np.inf], np.nan).dropna()

        ## vector of variables to be put in the initial value OLS
        
        self.variables = []
        
        if self.GMM_cons:
            self.variables.append('const')
        
        for key in self.vars_how:
            if self.vars_how[key][1] != 'no input':
                self.variables.append(key)
                
        if self.translog == True: # include the translog function specification
            
            var_list_support = [] # list needed for interaction terms
            for key in self.vars_how:
                if self.vars_how[key][1] != 'no input':
                    var_list_support.append(key)
            
            for key in self.vars_how:
                
                if self.vars_how[key][1] != 'no input':
                    self.variables.append(key+'2')
                    
                    if self.tl_inter: # construct interaction terms
                        for key2 in var_list_support: 
                            if key2 != key:
                                self.variables.append(key+key2)
                                
                        var_list_support.remove(key) # drop variable from list, so as not to duplicate terms


        
        Y = self.dta[['y']]
    
        X = self.dta[self.variables+self.init_control+self.FE] # the FE variables will get removed afterwards
        
        nb_var = X.shape[1] -len(self.init_control) - len(self.FE) # getting the number of variables

        if self.init_FE:
            if self.FE_demean:
                X = X.drop(columns=self.FE)
                if self.FE != []: # demean for fixed effects
                    if len(self.FE)<3:
                        for var_FE in self.FE:
                            X = X - X.mean(level=var_FE) # demean the variable
                            Y = Y - Y.mean(level=var_FE) # demean the variable
                    else: print('Impossible to demean with more than 2 FE levels')
            else:
                if self.FE != []: # add desired fixed effects
                    if self.GMM_cons:
                        dummies = pd.get_dummies( sum(X[self.FE].values.tolist(),[]) , drop_first=True )
                    else:
                        dummies = pd.get_dummies( sum(X[self.FE].values.tolist(),[]) , drop_first=False )
                    X = pd.concat([X.reset_index(drop=True),dummies.reset_index(drop=True)], axis=1)
                    Y = Y.reset_index(drop=True)
                    X = X.drop(columns=self.FE)


        C=pd.concat([X,Y],axis=1).dropna() # drop missing values
        X = C.drop(columns=['y']); Y = C[['y']] # get back X and Y

        X = np.array(X)
        Y = np.array(Y)

        if self.demean:
            Y = Y - np.nanmean(Y)
            if self.init_FE and self.FE_demean==False:
                if self.GMM_cons:
                    X[:,1:len(self.interlist)] = X[:,1:len(self.interlist)] - np.nanmean(X[:,1:len(self.interlist)], axis=0)
                else:
                    X[:,0:len(self.interlist)] = X[:,0:len(self.interlist)] - np.nanmean(X[:,0:len(self.interlist)], axis=0)
            elif self.FE_demean:
                pass
            else:
                X = X - np.nanmean(X, axis=0)


        b_ols = np.linalg.pinv(X)  @ Y 
        self.b_ols = np.squeeze(b_ols[:nb_var,])
            
        # substitute the initial values specified manually in beta_init
        self.b_ols = np.array( list(self.beta_init)  + list(self.b_ols[len(self.beta_init):]) ) 

        if self.verbose and len(self.beta_init)==0:
            print('Initial values computed with OLS:')
            print(self.b_ols)
        if self.verbose and len(self.beta_init)>0 and len(self.beta_init)<len(self.variables):
            print('Initial values:')
            print(self.b_ols)
            print('Out of which the following were imputed:')
            print(self.beta_init)
        if self.verbose and len(self.beta_init)==len(self.variables):
            print('Imputed Initial values:')
            print(self.beta_init)

    #----------------------------------------------------------#
    def initialize_GMM(self):
        """
        Construct the matrices that are fed to the GMM
        nonlinear solver.
        Note that whether an input is static or 
        dynamic impacts on the timing of the moment condition.
        """	 
        ## Construct matrices for GMM
        
        # construct the matrix of instruments
        # Keep an eye on the right moment condition
        instruments = []
        
        if self.GMM_cons:
            instruments.append('const')
        
        for key in self.vars_how:
            if self.vars_how[key][1] == 'static':
                instruments.append(key + '_lag')
            elif self.vars_how[key][1] == 'dynamic':
                instruments.append(key)
            else:
                pass
        
        if self.translog == True:
            
            var_list_support = [] # list needed for interaction terms
            for key in self.vars_how:
                if self.vars_how[key][1] != 'no input':
                    var_list_support.append(key)
                    
            for key in self.vars_how:
                
                if self.vars_how[key][1] == 'static':
                    instruments.append(key + '_lag2')
                elif self.vars_how[key][1] == 'dynamic':
                    instruments.append(key + '2')
                else:
                    pass
                
                if self.tl_inter and self.vars_how[key][1] != 'no input': # construct interaction terms
                    for key2 in var_list_support: 
                        if key2 != key:
                            
                            if self.vars_how[key][1] == 'static':
                                first = key + '_lag'
                            elif self.vars_how[key][1] == 'dynamic':
                                first = key
                            else:
                                pass
                            
                            if self.vars_how[key2][1] == 'static':
                                second = key2 + '_lag'
                            elif self.vars_how[key2][1] == 'dynamic':
                                second = key2
                            else:
                                pass
                            
                            instruments.append(first + second)
                            
                    var_list_support.remove(key) # drop variable from list, so as not to duplicate terms
                    
        self.Z = np.array(self.dta[instruments])
        
        if self.demean:
            if self.GMM_cons:
                self.Z[:,1:] = self.Z[:,1:] - np.nanmean(self.Z[:,1:], axis=0)
            else:
                self.Z = self.Z - np.nanmean(self.Z, axis=0)
        
        ## vector of variables in the production function estimated with GMM
        
        self.variables = []
        lagged_variables = []
        
        if self.GMM_cons:
            self.variables.append('const')
            lagged_variables.append('const')
        
        
        for key in self.vars_how:
            if self.vars_how[key][1] != 'no input':
                self.variables.append(key)
                lagged_variables.append(key+'_lag')
                
        if self.translog == True: # include the translog function specification
            
            var_list_support = [] # list needed for interaction terms
            for key in self.vars_how:
                if self.vars_how[key][1] != 'no input':
                    var_list_support.append(key)
            
            for key in self.vars_how:
                
                if self.vars_how[key][1] != 'no input':
                    self.variables.append(key+'2')
                    lagged_variables.append(key+'_lag2')
                    
                    if self.tl_inter: # construct interaction terms
                        for key2 in var_list_support: 
                            if key2 != key:
                                self.variables.append(key+key2)
                                lagged_variables.append(key+'_lag'+key2+'_lag')
                                
                        var_list_support.remove(key) # drop variable from list, so as not to duplicate terms

        self.X = np.array(self.dta[self.variables])
        self.X_lag = np.array(self.dta[lagged_variables])
        
        # dependent variable for initial OLS values
        self.Y = np.array([np.array(self.dta['y'])]).T
        
        # constant
        self.C = np.array([np.array(self.dta['const'])]).T
        # dependent variable of the GMM
        self.phi = np.array([np.array(self.dta['phi'])]).T
        self.phi_lag = np.array([np.array(self.dta['phi_lag'])]).T
                                
        if self.demean:
            if self.GMM_cons:
                self.X[:,1:] = self.X[:,1:] - np.nanmean(self.X[:,1:], axis=0)
                self.X_lag[:,1:] = self.X_lag[:,1:] - np.nanmean(self.X_lag[:,1:], axis=0)
            else:
                self.X = self.X - np.nanmean(self.X, axis=0)
                self.X_lag = self.X_lag - np.nanmean(self.X_lag, axis=0)
            self.phi = self.phi - np.nanmean(self.phi, axis=0)
            self.phi_lag = self.phi_lag - np.nanmean(self.phi_lag, axis=0)
        
    
    #----------------------------------------------------------# 
    def moments(self, betas_1D):
        """
        This builds the GMM objective function that needs be minimized.
        """
        betas = np.array([betas_1D]).T
                        
        omega = self.phi - self.X @ betas
        omega_lag = self.phi_lag - self.X_lag @ betas
                
        omega_lag_pol = omega_lag
        if self.AR_c: # include the constant
            omega_lag_pol = np.concatenate((self.C,omega_lag_pol),axis=1) # adding the constant to AR
        if self.AR_sqlag: # include the squared lag
            omega_lag_pol = np.concatenate((omega_lag_pol,np.square(omega_lag)),axis=1) # adding squared lag to AR
        
        g_b = np.linalg.inv(omega_lag_pol.T @ omega_lag_pol) @ (omega_lag_pol.T @ omega)

        self.g_b = g_b #Save the AR process parameters (can be accessed from the class)
        
        xi = omega - omega_lag_pol @ g_b
        
        # define the moments
        if self.NormMoments: # normalize moments so as to have correlation, rather than convariance
            moments = (self.Z.T @ xi) / ((np.array([np.diagonal(self.Z.T @ self.Z)]).T)**0.5 * (xi.T @ xi)**0.5)  
        else: # leave moments as covariance
            moments = self.Z.T @ xi
            
        # format the moments for the minimizer or the solver    
        if self.optim=='FSOLVE': # FSOLVE requires one value to bring to 0 per each moment (a vector)
            return np.squeeze( moments )
        else: # other solvers require just one positive value to minimize
            if self.LogMoments:
                return np.log( np.squeeze( moments.T @ moments ) )
            else:
                return np.squeeze( moments.T @ moments )
    
    
    #----------------------------------------------------------# 
    def fit_GMM(self):
        """
        Brings self.moments() to 0 by using three possible optimizers.
        """
        t0 = time()
        
        self.optim_used = str(self.optim)
        
        ##Choose the optminzers
        if self.optim=='NM':
            ##Nelder-Mead simplex algorithm
            #NB: robust and good for large sample
                       
            # with LogMoments the function goes much faster to the minimum, but actually it never reaches it (it is -inf).
            # therefore with LogMoments the optimizer is raising errors, even when it is working
            if self.LogMoments: maxfun = 10000
            else: maxfun = 10000  
                
            optimizer = optimize.fmin
            self.conv = optimizer( self.moments, self.b_ols, initial_simplex=None, full_output=True,disp=self.verbose,xtol=10**(-6),ftol=10**(-7),maxfun=maxfun) #this is for fmin
            
            if self.verbose:
                print('Fun Value: ' + str(self.conv[1]))
                

        elif self.optim=='BFGS':
            ##quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
            #NB: bfgs algo (is good and precise on small sample but memory consuming not so good for large sample)
            optimizer = optimize.fmin_bfgs
            self.conv = optimizer( self.moments, self.b_ols, full_output=True,disp=self.verbose,gtol=10**(-6)) #this is for fmin_bfgs
            
            if self.verbose:
                print('Fun Value: ' + str(self.conv[1]))
                
        elif self.optim=='FSOLVE':
            ##  wrapper around MINPACK’s hybrd and hybrj algorithms. 
            #NB: Find roots of a non linear equation, which is faster in most applications
            optimizer = optimize.fsolve
            self.conv = optimizer( self.moments, self.b_ols, full_output=True) #this is for fsolve
            if self.verbose:
                print(self.conv[-1])
                self.optim = 'NM'; print( 'Fun Value: ' + str(self.moments(self.conv[0])) ); self.optim = 'FSOLVE' #compute the value of the criteria for comparaison
                print('        fvec: ' + str(self.conv[-3]['fvec']))
                print('        nfev: ' + str(self.conv[-3]['nfev']))
                
        elif self.optim=='BASINHOPPING':
            ##Find the global minimum of a function using the basin-hopping algorithm
            optimizer = optimize.basinhopping
            ret= optimizer( self.moments, self.b_ols, disp=False ) #this is for basinhopping
            self.betas = ret.x
            self.ret=ret

            if self.verbose:
                print( ret.message[0] )
                print('Fun Value: ' + str(ret.fun))        
        
        elif self.optim=='Iter':
            
            not_converged = True # used to keep track of the procedure
            
            ## Try to run NM
            
            if self.verbose:
                print('')
                print('Attempting NM:')
        
            initial_simplex_our = None
                
            optimizer = optimize.fmin
            self.conv = optimizer( self.moments, self.b_ols, initial_simplex=initial_simplex_our, full_output=True,disp=self.verbose,xtol=10**(-6),ftol=10**(-7),maxfun=10000) #this is for fmin
            
            if self.verbose:
                print('Fun Value: ' + str(self.conv[1]))
            
            if self.conv[4] == 0:
                not_converged = False
                self.optim_used = 'NM'
            
            ## Try to run FSOLVE if NM did not converge
            
            if not_converged:

                if self.verbose:
                    print('')
                    print('Attempting FSOLVE:')
                    
                optimizer = optimize.fsolve
                self.optim = 'FSOLVE'
                self.conv = optimizer( self.moments, self.b_ols, full_output=True) #this is for fsolve
                self.optim = 'Iter'
                if self.verbose:
                    print(self.conv[-1])
                    self.optim = 'NM'; print( 'Fun Value: ' + str(self.moments(self.conv[0])) ); self.optim = 'FSOLVE' #compute the value of the criteria for comparaison
                    print('        fvec: ' + str(self.conv[-3]['fvec']))
                    print('        nfev: ' + str(self.conv[-3]['nfev']))
                    
                if self.conv[-2] == 1:
                    not_converged = False
                    self.optim_used = 'FSOLVE'
                    
            ## Try to run BASINHOPPING
            
            if not_converged:

                if self.verbose:
                    print('')
                    print('Attempting BASINHOPPING:')
                    
                optimizer = optimize.basinhopping
            
                ret= optimizer( self.moments, self.b_ols, disp=False ) #this is for basinhopping
                self.betas = ret.x
                self.ret=ret
    
                if self.verbose:
                    print( ret.message[0] )
                    print('Fun Value: ' + str(ret.fun))
                    
                if self.ret.lowest_optimization_result.success == True:
                    not_converged = False
                else:
                    if self.verbose:
                        print('')
                        print('CONVERGENCE NOT ACHIEVED!')
                        print('')
                
                self.optim_used = 'BASINHOPPING'
                
            if self.verbose:
                print('') # do it to make the output look good
                
                
                
        t1 = time()
        
        # the coefficients betas
        if self.optim_used=='BASINHOPPING':
            pass
        elif self.optim_used=='SLSQP' or self.optim_used=='differential_evolution':
            self.betas=self.conv['x']
        else:
            self.betas=self.conv[0]
                        
        if self.verbose:
            print('Fitting time: ' + str(np.round(t1-t0,2)) + ' s')
            print('Resulting Betas:')
            print(self.betas)
        
        return self.betas
    

    #----------------------------------------------------------#
    def compute_ar1(self):
        """
        Recover AR(1) parameters for productivity (omega) at the final GMM betas.
        Replicates the AR step inside moments() at the solution.

        Stores
        ------
        self.rho     : float, AR(1) persistence coefficient
        self.sigma_xi: float, std dev of the AR(1) innovation xi
        """
        betas = np.array([self.betas]).T

        omega     = self.phi     - self.X     @ betas
        omega_lag = self.phi_lag - self.X_lag @ betas

        omega_lag_pol = omega_lag
        if self.AR_c:
            omega_lag_pol = np.concatenate((self.C, omega_lag_pol), axis=1)
        if self.AR_sqlag:
            omega_lag_pol = np.concatenate((omega_lag_pol, np.square(omega_lag)), axis=1)

        g_b = np.linalg.inv(omega_lag_pol.T @ omega_lag_pol) @ (omega_lag_pol.T @ omega)
        xi  = omega - omega_lag_pol @ g_b

        # rho is the coefficient on omega_lag (first column if no constant, second if constant)
        rho_idx    = 1 if self.AR_c else 0
        self.rho      = float(g_b[rho_idx, 0])
        self.sigma_xi = float(np.std(xi))

        if self.verbose:
            print('AR(1) rho = '      + str(np.round(self.rho,      4)))
            print('AR(1) sigma_xi = ' + str(np.round(self.sigma_xi, 4)))

    #----------------------------------------------------------#
    def compute_markups(self):
        """
        Calculate markups given estimated elasticities. Note that
        there is going to be one markup per variable input.
        """
        
        ## construct the elasticity
        
        self.dta_mu['omega'] = self.dta_mu['phi'] - self.dta_mu[self.variables].values @ self.betas

        if self.sales_correction:
            sales_corr_term = np.exp(self.dta_mu['epsilon']);
        else:
            sales_corr_term = 1;
        
        for key in self.vars_how:
            if self.vars_how[key][2] == 'variable':
                
                elasticity = self.betas[self.variables.index(key)]
                
                if self.translog: # add squared terms in case of translog
                    elasticity = elasticity + 2*self.betas[self.variables.index(key+'2')]*self.dta_mu[key]
                    
                    if self.tl_inter: # add interaction term in case of translog
                        for key2 in self.vars_how:
                            if self.vars_how[key2][1] != 'no input' and key2 != key:
                                try: # needed if the vars_how is not an ordered dict
                                    elasticity = elasticity + self.betas[self.variables.index(key+key2)]*self.dta_mu[key2]
                                except:
                                    elasticity = elasticity + self.betas[self.variables.index(key2+key)]*self.dta_mu[key2]

                self.dta_mu['mu_' + key] = elasticity * (self.dta_mu['sale']/sales_corr_term) / np.exp(self.dta_mu[key])
        
                      
    #----------------------------------------------------------# 
    def clean_dta_mu(self):
        """
        Clean the output table to get nice results.
        """
        self.dta_mu = self.dta_mu.drop(columns=['date'])
        self.dta_mu = self.dta_mu.reset_index()

        
        vars_keep = ['date','firmid','sale','varcost','sector','epsilon','omega']
        for key in self.vars_how:
            if self.vars_how[key][2] == 'variable':
                vars_keep.append('mu_' + key)
                
#        self.dta_mu = self.dta_mu[vars_keep]
        
        if self.verbose:
            for key in self.vars_how:
                if self.vars_how[key][2] == 'variable':
                    mu = 'mu_' + key
                    median_mu = np.round( np.median(self.dta_mu[mu].dropna()) , 3)
                    iqr_mu = np.round( ( np.quantile(self.dta_mu[mu].dropna(),0.75) - np.quantile(self.dta_mu[mu].dropna(),0.25) ) / np.quantile(self.dta_mu[mu].dropna(),0.5)    ,3)
                    
                    print('Variable {} has {} median and {} iqr markup'.format(key, median_mu, iqr_mu))

    
