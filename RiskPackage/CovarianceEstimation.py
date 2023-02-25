import pandas as pd
import numpy as np

'''
===============================================================================================================
Exponentially Weighted Covariance
===============================================================================================================
'''

#  Exponentially Weighted Moving Average for Volatility Model -- Exponentially Weighted Covariance
class EWMA:
    """
    Calculate the Exponentially Weighted Covariance & Correaltion Matrix
    
    Parameter: 
        data (Array)  -- return data for calculating Covariance & Correaltion Matrix (array)
        lambda_(Float)  -- smoothing parameter (less than 1)
        flag (Boolean) -- a flag (optional) to dertermine whether to subtract mean from data.
                            if it set False, data would not subtract its mean.

    fomula: \sigma_t^2=\lambda \sigma_{t-1}^2+(1-\lambda)r_{t-1}^2

    Usage:  
        model=EWMA(data,0.97)
        cov_mat=model.cov_mat
        corr_mat=model.corr_mat
    """
    # initialization 
    def __init__(self,data,lambda_,flag=False):
        self.__data=data if flag==False else data-data.mean(axis=0)
        self.__lambda=lambda_
        self.get_weight() 
        self.cov_matrix()
        self.corr_matrix()

    # calculate the weight matrix
    def get_weight(self):
        n=self.__data.shape[0]
        weight_mat=[(1-self.__lambda)*self.__lambda**(n-i-1) for i in range(n)]
        self.__weight_mat=np.diag(weight_mat)

    # calculate cov_matrix
    def cov_matrix(self):
        self.__cov_mat=self.__data.T @ self.__weight_mat @ self.__data

    # calculate corr_matrix
    def corr_matrix(self):
        n=self.__data.shape[1]
        invSD=np.sqrt(1./np.diag(self.__cov_mat))
        invSD=np.diag(invSD)
        self.__corr_mat=invSD @ self.__cov_mat @ invSD
        return self.__corr_mat

    # plot the cumulative weight
    def plot_weight(self,k=None,ax=None,label=None):
        weight=np.diag(self.__weight_mat)[::-1]
        cum_weight=weight.cumsum()/weight.sum()
        sns.lineplot(cum_weight,ax=ax,label="{:.2f}".format(label) if label!=None else "")
        if ax!=None:
            ax.set_xlabel('Time Lags')
            ax.set_ylabel('Cumulative Weights')
            ax.set_title("Weights of differnent lambda")
        ax.legend(loc='best')

    @property
    def cov_mat(self):
        return self.__cov_mat    

    @property
    def corr_mat(self):
        return self.__corr_mat


'''
===============================================================================================================
Covariance Generator
===============================================================================================================
'''

class Cov_Generator:
    """
    Convariance Derivation through differnet combination of EW covariance, EW correlation, covariance and correlation.

    Parameter:
        data(Array) -- data which is used for get the EW covariance, EW correlation, covariance and correlation
    
    Usage:
        cov_generator=Cov_Generator(data)
        cov_generator.EW_cov_corr()
        cov_generator.EW_corr_cov()
        cov_generator.EW_corr_EW_cov()
        cov_generator.corr_cov()
    """
    def __init__(self,data):
        self.__data = data
        self.__EWMA = EWMA(data,0.97)
        self.__EW_cov = self.__EWMA.cov_mat
        self.__EW_corr = self.__EWMA.corr_mat
        self.__cov = np.cov(data.T)
        invSD=np.diag(1/np.sqrt(np.diag(self.__cov)))
        self.__corr = invSD @ self.__cov @ invSD

    # EW_cov + corr
    def EW_cov_corr(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__corr @ std

    # EW_corr + cov
    def EW_corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__EW_corr @ std

    # EW_corr + EW_cov
    def EW_corr_EW_cov(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__EW_corr @ std

    # corr + cov
    def corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__corr @ std