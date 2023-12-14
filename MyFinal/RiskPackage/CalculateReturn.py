import pandas as pd
import numpy as np
'''
===============================================================================================================
Calculate Return
===============================================================================================================
'''

def return_calculate(Price,option="DISCRETE",rm_means=True):
    '''
        Provide two ways to calculate the return from Price dataframce.
    '''
    # calculate the log normal return 
    if option == 'CONTINUOUS':
        returns = np.log(Price/Price.shift()).dropna()
    # calculate the discrete return 
    elif  option == 'DISCRETE':
        returns = Price.pct_change().dropna()
    # other undefined option will cause error
    else:
        raise Exception("Unknown Option!")
    # remove mean from the returns
    return returns if rm_means==False else returns-returns.mean()