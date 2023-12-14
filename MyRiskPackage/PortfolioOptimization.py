
from scipy import optimize
import pandas as pd
import numpy as np


def super_efficient_portfolio(expected_rts,cov,rf=0.0425):
    '''Given a target return, use assets to find the optimal portfolio with lowest risk'''
    fun=lambda wts: -(wts@expected_rts-rf)/np.sqrt(wts@cov@wts)
    x0 = np.full(expected_rts.shape[0],1/expected_rts.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(expected_rts.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res

def RiskParity(cov):
    '''Given a target return, use assets to find the optimal portfolio with lowest risk'''
    fun=lambda w: (w*(cov@w)/np.sqrt(w@cov@w)).std()
    x0 = np.full(cov.shape[0],1/cov.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(cov.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res
    
def riskBudget(w,cov):
    '''Calculate the portion of risk each stock of portfolio has. The sum of result is 1'''
    portfolioStd=np.sqrt(w@cov@w)
    Csd=w*(cov@w)/portfolioStd
    return Csd/portfolioStd