import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import math
import itertools
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
sys.path.append("..")
from RiskPackage.CalculateReturn import return_calculate
from RiskPackage.RiskMetrics import RiskMetrics
from scipy import linalg


def gbsm(s,strike,ttm,vol,rf,c,call=True):
    '''
    Generalize Black Scholes Merton
    rf = c       -- Black Scholes 1973
    c = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
    c = 0        -- Black 1976 futures option model
    c,r = 0      -- Asay 1982 margined futures option model
    c = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency

    Option valuation via BSM closed formula
    European Style.  Assumed LogNormal Prices
    s - Underlying Price
    strike - Strike Price
    ttm - time to maturity
    rf - Risk free rate
    vol - Yearly Volatility
    c - Cost of Carry
    call - Call valuation if set True
    '''
    d1=(np.log(s/strike)+(c+vol**2/2)*ttm)/vol/np.sqrt(ttm)
    d2=d1-vol*np.sqrt(ttm)
    if call:
        return s*np.exp((c-rf)*ttm)*norm.cdf(d1)-strike*np.exp(-rf*ttm)*norm.cdf(d2)
    else:
        return strike*np.exp(-rf*ttm)*norm.cdf(-d2)-s*np.exp((c-rf)*ttm)*norm.cdf(-d1)

        
def greeks_closed_form(s,strike,ttm,vol,rf,c,call=True):
    '''Closed from for greeks calculation from Generalize Black Scholes Merton
        Generalize Black Scholes Merton:
        rf = c       -- Black Scholes 1973
        c = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
        c = 0        -- Black 1976 futures option model
        c,r = 0      -- Asay 1982 margined futures option model
        c = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency

        Option valuation via BSM closed formula
        European Style.  Assumed LogNormal Prices
        s - Underlying Price
        strike - Strike Price
        ttm - time to maturity
        rf - Risk free rate
        vol - Yearly Volatility
        c - Cost of Carry
        call - Call valuation if set True
    '''
    d1=(np.log(s/strike)+(c+vol**2/2)*ttm)/vol/np.sqrt(ttm)
    d2=d1-vol*np.sqrt(ttm)
    optionType=['Call'] if call else ['Put']
    ans=pd.DataFrame(index=optionType,columns=['Detla','Gamma','Vega','Theta','Rho','Carry Rho'])
    if call:
        ans['Detla'] = np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)
        ans['Theta'] = -s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*vol/(2*np.sqrt(ttm))-(c-rf)*s*np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)-rf*strike*np.exp(-rf*ttm)*norm.cdf(d2,loc=0,scale=1)
        # ans['Rho'] = ttm*strike*np.exp(-rf*ttm)*norm.cdf(d2,loc=0,scale=1) - s*ttm*np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)
        ans['Rho'] = ttm*strike*np.exp(-rf*ttm)*norm.cdf(d2,loc=0,scale=1)

        ans['Carry Rho'] = ttm*s*np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)
    else:
        ans['Detla'] = np.exp((c-rf)*ttm)*(norm.cdf(d1,loc=0,scale=1)-1)
        ans['Theta'] = -s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*vol/(2*np.sqrt(ttm))+(c-rf)*s*np.exp((c-rf)*ttm)*norm.cdf(-d1,loc=0,scale=1)+rf*strike*np.exp(-rf*ttm)*norm.cdf(-d2,loc=0,scale=1)
        ans['Rho'] = -ttm*strike*np.exp(-rf*ttm)*norm.cdf(-d2,loc=0,scale=1)
        # ans['Rho'] = -ttm*strike*np.exp(-rf*ttm)*norm.cdf(-d2,loc=0,scale=1)+ttm*s*norm.cdf(-d1,loc=0,scale=1)*exp((b-rf)*ttm)
        ans['Carry Rho'] = -ttm*s*np.exp((c-rf)*ttm)*norm.cdf(-d1,loc=0,scale=1)
    ans['Gamma'] = norm.pdf(d1,loc=0,scale=1)*np.exp((c-rf)*ttm)/(s*vol*np.sqrt(ttm))
    ans['Vega'] = s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*np.sqrt(ttm)

    return ans


