import pandas as pd
import numpy as np
import bisect
import time
from RiskPackage.NonPsdFixes import chol_psd
import matplotlib.pyplot as plt
from RiskPackage.NonPsdFixes import Weighted_F_norm
import seaborn as sns
sns.set_theme()

'''
===============================================================================================================
 Principal component analysis (PCA)
===============================================================================================================
'''

class PCA:
    """
    Reducing the dimensionality of the dataset.
    
    Parameter:
        cov_mat (Array) -- covarinance matrix for dimensionality reduction
        threshold (Float)(optional) -- the threshold of cumulative variance explained
    
    Usage:
        PCA_model=PCA(cov_mat,threshold)
        PCA_model.plot()
        PCA_model.pct_explain_val_vec()
    """
    # initialization
    def __init__(self,cov_mat,threshold=None):
        self.__cov_mat=cov_mat
        self.__threshold=threshold
        self.run()
    
    # Conduct the PCA
    def run(self):
        self.__eig_val,self.__eig_vec = np.linalg.eigh(self.__cov_mat)
        # pick the eigenvalues which is bigger than 0 and corresponding eigenvector   
        idx=self.__eig_val>1e-8
        self.__eig_val = self.__eig_val[idx]
        self.__eig_vec = self.__eig_vec [:,idx]       
        # sort since the result given by numpy is lowest to highest, flip them up
        sorted_indices = np.argsort(self.__eig_val)
        self.__eig_val=self.__eig_val[sorted_indices[::-1]]
        self.__eig_vec=self.__eig_vec[:,sorted_indices[::-1]]
        

    # calculate the cumulative percent of variance explained
    def percent_expalined(self,k=None):
        k = self.__eig_val.shape[0] if k==None else k
        k_eig_val=self.__eig_val[:k]

        return k_eig_val.cumsum()/k_eig_val.sum()

    # plot the cumulative percent of variance explained
    def plot(self,k=None,ax=None,label=None):
        explain=self.percent_expalined(k)
        sns.lineplot(explain,ax=ax,label="{:.2f}".format(label) if label!=None else "" )
        if ax!=None:
            ax.set_xlabel('Number of Component')
            ax.set_ylabel('Cumulative Variance Explained')
            ax.set_title("Cumulative Variance Explained via different lambda")
        ax.legend(loc='best')
    
    # Given the threshold, calculate the needed eigenvalues and corresponding eigenvectors
    def pct_explain_val_vec(self):
        eig_val=self.__eig_val
        pct_cum_var=eig_val.cumsum()/eig_val.sum()
        pct_cum_var[-1]=1
        # if cumulative percent of variance explained is larger than threshold, then break
        k = bisect.bisect_left(pct_cum_var, self.__threshold) 
        return self.__eig_val[:k+1],self.__eig_vec[:,:k+1]

'''
===============================================================================================================
Simulator (Direct Simulation & PCA Simulation)
===============================================================================================================
'''

class Simulator(Weighted_F_norm):
    """
    Simulator for DirectSimulation & PCA_Simulation

    Parameter: 
        cov_mat (Array) -- covariance matrix
        draw_num (Float) -- the number of sample draw from simulation
    
    Usage:
        simulator = Simulator(cov_mat,draw_num)
        simulator.DirectSimulation()
        simulator.PCA_Simulation(pct)
    """

    def __init__(self,cov_mat,draw_num):
        self.__cov_mat=cov_mat
        self.__draw_num=draw_num
        self.__Direct_run_time=None
        self.__PCA_run_time=None
        self.__X=None # generated sample

    def DirectSimulation(self):
        """Cholesky"""
        t = time.time()
        root=chol_psd(self.__cov_mat).root
        n=root.shape[0]
        rand_norm=np.random.normal(0, 1, size=(n, self.__draw_num))
        X= root @ rand_norm
        self.__Direct_run_time = time.time()-t
        self.__X=X #simulated sample
        return X

    def PCA_Simulation(self,threshold):
        """PCA"""
        t = time.time()
        PCA_model=PCA(self.__cov_mat,threshold=threshold)
        eig_val,eig_vec=PCA_model.pct_explain_val_vec()
        B= eig_vec @ np.diag(np.sqrt(eig_val))
        n=B.shape[1]
        rand_norm=np.random.normal(0, 1, size=(n, self.__draw_num))
        X= B @ rand_norm
        self.__PCA_run_time = time.time()-t
        self.__X=X #simulated sample
        return X

    # get the F norm of difference between covariance of simulation and original covariance
    def err_F_norm(self):
        n=self.__cov_mat.shape[0]
        w=np.eye(n)
        return self.compare_F(self.__cov_mat,np.cov(self.__X),w)
     
    @property
    def Direct_run_time(self):
        return self.__Direct_run_time

    @property
    def PCA_run_time(self):
        return self.__PCA_run_time