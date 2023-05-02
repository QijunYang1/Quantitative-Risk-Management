
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.integrate import quad
sns.set_theme()

'''
    Introduction:
        FittedModel -- Fitted Distribution Prototype is the virtual class which can not be directly used for model fitting.
        
        The specific class below implement the FittedModel class:
            1. Fitted Normal Distribution
            2. Fitted Fit Generalized T Distribution
            3. Fitted Fit Generalized T Distribution
        We could use those to fit the data, but keep in mind the data should be np.array format.

        If we have a dataframe which contains many companies' return data (just an example), I write another class, ModelFitter,
        which can fit the such dataframe with a given distribution. 
        It will return an 1-D dataframe of distributions fitted using the data of each column.
        -- The reason why I write it is I want to have a chain of same distributions but each distribution may have different 
        parameters. Therefore, I could use them to do the copula simulation.
'''

'''
===============================================================================================================
Fitted Distribution Prototype
===============================================================================================================
'''

class FittedModel:
    '''The prototype of fitted distribution.'''
    def __init__(self):
        self.dist=self.set_dist() # the distribution
        self.frz_dist=None # the distribution which has specific parameters

    def set_dist(self):
        '''Need to be implemented in subclass to set the dist.'''
        raise NotImplementedError
    
    def freeze_dist(self,parameters):
        '''Need to be implemented in subclass to set the parameters of different distribution.'''
        raise NotImplementedError

    def fit(self,data,x0,cons):
        '''
        Use MLE to fit the distribution
        x0 is initial paremeters which needed to be implemented in subclass
        cons is constraints of parameters which needed to be implemented in subclass
        '''
        def nll(parameters,x):
            '''Negative likelihood function'''
            self.freeze_dist(parameters)
            ll=self.frz_dist.logpdf(x=x).sum()
            return -ll
        MLE = minimize(nll, x0=x0, args=data, constraints = cons) # MLE 
        self.freeze_dist(MLE.x)
        self.fitted_parameters=MLE.x

    @property
    def fitted_dist(self):
        return self.frz_dist

'''
===========================================
Fitted Normal Distribution
===========================================
'''

class Norm(FittedModel):
    def set_dist(self):
        '''set the distribution to be normal'''
        return stats.norm
        
    def freeze_dist(self,parameters):
        '''set the parameters of norm: parameters[0]--mu, parameters[1]--std'''
        self.frz_dist=self.dist(loc=parameters[0],scale=parameters[1])

    def fit(self,data):
        '''set the initial parameters and cons to call the father's fit'''
        x0 = (data.mean(),data.std())  # initial paremeters
        cons = [ {'type':'ineq', 'fun':lambda x:x[1]} ] # standard deviation is non-negative
        super().fit(data,x0,cons)

'''
===========================================
Fitted Fit Generalized T Distribution
===========================================
'''

class T(FittedModel):
    def set_dist(self):
        '''set the distribution to be normal'''
        return stats.t
        
    def freeze_dist(self,parameters):
        '''set the parameters of norm: parameters[0]--degree of freedom, parameters[1]--mu, parameters[2]--std'''
        self.frz_dist=self.dist(df=parameters[0],loc=parameters[1],scale=parameters[2])
        
    def fit(self,data):
        '''set the initial parameters and cons to call the father's fit'''
        # degree of freedom of t should be greater than 2; standard deviation is non-negative
        cons=[ {'type':'ineq', 'fun':lambda x:x[0]-2} , {'type':'ineq', 'fun':lambda x:x[2]} ] 
        mu=data.mean()
        df=6/stats.kurtosis(data,bias=False)+4
        df = 2.5 if df<=2 else df
        std=np.sqrt(data.var()*df/(df-2))
        x0 = np.array([df,mu,std]) # initial parameter
        super().fit(data,x0,cons)

'''
===========================================
Fitted Fit Generalized T Distribution (Mean 0)
===========================================
'''

class T_mean0(FittedModel):
    def set_dist(self):
        '''set the distribution to be normal'''
        return stats.t
        
    def freeze_dist(self,parameters):
        '''set the parameters of norm: parameters[0]--degree of freedom, parameters[1]--mu, parameters[2]--std'''
        self.frz_dist=self.dist(df=parameters[0],loc=parameters[1],scale=parameters[2])
        
    def fit(self,data):
        '''set the initial parameters and cons to call the father's fit'''
        # degree of freedom of t should be greater than 2; standard deviation is non-negative
        cons=[ {'type':'ineq', 'fun':lambda x:x[0]-2} , {'type':'eq', 'fun':lambda x:x[1]},{'type':'ineq', 'fun':lambda x:x[2]} ] 
        mu=data.mean()
        df=6/stats.kurtosis(data,bias=False)+4
        df = 2.5 if df<=2 else df
        std=np.sqrt(data.var()*df/(df-2))
        x0 = np.array([df,mu,std]) # initial parameter
        super().fit(data,x0,cons)

'''
===============================================================================================================
Model Fitter
===============================================================================================================
'''

class ModelFitter:
    ''' Fit the data with Model, return a group of fitted distributions

        Parameters:
            FittedModel(Class) ---- a subclass of FittedModel class

        Usage:
            dists=ModelFitter(FittedModel).fit(data)
    '''

    def __init__(self,FittedModel):
        ''' Initialize the model within the class to fit all the data.'''
        self.model=FittedModel()
    
    def fit(self,data):
        '''Fit all the data with the model inside the Fitter
            Data(Dataframe) -- return of stock
        '''
        dists=[]
        for name in data.columns:
            rt=np.array(data[name].values)
            self.model.fit(rt)
            dists.append(self.model.fitted_dist)
        dists=pd.DataFrame(dists).T
        dists.columns=data.columns
        dists.index=["distribution"]
        return dists