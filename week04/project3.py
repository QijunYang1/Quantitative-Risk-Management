import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import seaborn.objects as so
import bisect
sns.set_theme()

#  Exponentially Weighted Moving Average for Volatility Model -- Exponentially Weighted Covariance
class EWMA:
    """
    Calculate the Exponentially Weighted Covariance & Correaltion Matrix
    
    Parameter: 
        data   -- data for calculating Covariance & Correaltion Matrix
        lambda_  -- smoothing parameter (less than 1)
        flag (optional)  -- a flag to dertermine whether to subtract mean from data.
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
        if len(self.__data.shape)==2:
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





class PCA:
    """
    Reducing the dimensionality of the dataset.
    
    Parameter:
        cov_mat -- covarinance matrix for dimensionality reduction
        threshold(optional) -- the threshold of cumulative variance explained
    
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


class NotPsdError(Exception):
    ''' 
    Used for expection raise if the input matrix is not sysmetric positive definite 
    '''
    pass

class chol_psd():
    '''
    Cholesky Decompstion: Sysmetric Positive Definite matrix could use Cholesky 
    algorithm to fatorize the matrix to the product between a lower triangle matrix and
    upper triangle matrix

    Parameter:
        matrix  --  Sysmetric Positive Definite (or Positive Semi-definite) 
                    matrix needed to do Cholesky Factorization.
    
    Formula: 
        matrix=L*L.T

    Usage:
        Chol_model=chol_psd(matrix)
        root=Chol_model.root
    '''
    # initialization
    def __init__(self,matrix):
        self.__psd=matrix
        self.run()

    # Perform the Cholesky Factorization
    def run(self):
        n=self.__psd.shape[0]
        root=np.zeros([n,n])
        for i in range(n):
            # diagonal
            root[i][i] = self.__psd[i][i] - root[i][:i] @ root[i][:i].T
            root[i][i]=0 if 0>=root[i][i]>=-1e-8 else root[i][i]
            # if the diagonal element is less than -1e-8, it might not be PSD
            if root[i][i]<0:
                raise NotPsdError("Not PSD!")
            root[i][i]=np.sqrt(root[i][i])
            
            #below the diagonal
            # if diagonal element is zero, set the following element of that column to be zero too
            if root[i][i]==0:
                continue
            for j in range(i+1,n):
                root[j][i]=(self.__psd[j][i]-root[i,:i] @ root[j,:i])/root[i][i]
        self.__root=root
        self.__ispsd=True

    @property
    def root(self):
        return self.__root   
    
    @property
    def ispsd(self):
        return self.__ispsd


class Weighted_F_norm:
    '''
    Given the weight matrix, calculate the Weighted Frobenius Norm. (Assume it's diagonal)
    '''
    def compare_F(self,mat_a,mat_b,mat_w):
        '''Give two matrix, use Weighted Frobenius Norm to calculate how close they are'''
        err = mat_a-mat_b #difference
        weighted_err = np.sqrt(mat_w) @ err @ np.sqrt(mat_w) 
        w_F_norm = np.sqrt(np.square(weighted_err).sum())
        return w_F_norm
    
    def calculate_F(self,mat,mat_w):
        "Given one matrix, calculate its Weighted Frobenius Norm"
        weighted_err = np.sqrt(mat_w) @ mat @ np.sqrt(mat_w)
        w_F_norm = np.sqrt(np.square(weighted_err).sum())
        return w_F_norm


class NotSysmetricError(Exception):
    ''' 
    Used for expection raise if the input matrix is not sysmetric
    '''
    pass

class NegativeEigError(Exception):
    ''' 
    Used for expection raise if matrix has the negative eigvalue
    '''
    pass

class PSD:
    """
    PSD class is used for Positive Semi-Definite Matrix confirmation.
    """
    @staticmethod
    def confirm(psd):
        # make sure sysmetric
        if not np.allclose(psd,psd.T):
            raise NotSysmetricError("Matrix does not equal to Matrix.T")
        # Make sure no negative eigenvalues
        eig_val=np.linalg.eigvals(psd)
        neg_eig=len(eig_val[eig_val<0])
        # No negative eigenvalues or Pass the Cholesky algorithm
        if neg_eig==0 or chol_psd(psd).ispsd:
            print("Matrix is Sysmetric Positive Definite!")
        else:
            raise NegativeEigError("Matrix has negative eigenvalue.")
        


class Non_psd_mat:
    """
    Used to generate the non-positive semi-definite matrix
    """
    def non_psd_mat(self,n):
        corr = np.full((n, n), 0.9)
        np.fill_diagonal(corr, 1)
        corr[0, 1] = 0.7357  
        corr[1, 0] = 0.7357 
        return corr



class near_psd(Weighted_F_norm):
    '''
    Rebonato and Jackel's Method to get acceptable PSD matrix 
    
    Parameters:
        not_psd -- the matrix which is not positive semi-definite matrix
        weight  -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)

    Usage:
        near_psd_model=near_psd(non_psd,weight)
        psd=near_psd_model.psd
    '''
    # initialization
    def __init__(self,not_psd,weight):
        self.__not_psd=not_psd
        self.__weight=weight
        self.run() # main function
        self.F_compare_norm(weight) # Weighted Frobenius Norm
        
    def run(self):
        n=self.__not_psd.shape[0]
        # Set the weight matrix to be identity matrix
        invSD = np.eye(n)
        corr=self.__not_psd
        # if the matrix is not correlation matrix, convert it to the correlation matrix
        if not np.allclose(np.diag(self.__not_psd),np.ones(n)):
            invSD=np.diag(1/np.sqrt(np.diag(self.__not_psd)))
            corr=invSD @ self.__not_psd @ invSD
        eig_val,eig_vec=np.linalg.eigh(corr) # eigenvalues & eigenvectors 
        eig_val[eig_val<0]=0 # adjust the negative value to 0
        # get the scale matrix
        scale_mat = np.diag(1/(eig_vec * eig_vec @ eig_val))
        B = np.sqrt(scale_mat) @ eig_vec @ np.sqrt(np.diag(eig_val))
        corr=B @ B.T
        # convert it back into original form
        SD=np.diag(1/np.diag(invSD))
        psd = SD @ corr @ SD
        self.__psd = psd

    # Weighted Frobenius Norm of the difference between near_psd and ono_psd
    def F_compare_norm(self,weight):
        self.__F = self.compare_F(self.__psd,self.__not_psd,weight)

    @property
    def psd(self):
        return self.__psd
    
    @property
    def F(self):
        return self.__F

class Higham_psd(Weighted_F_norm,chol_psd):
    '''
    Higham's Method to get nearest PSD matrix under the measure of Weighted Frobenius Norm
    
    Parameters:
        not_psd -- the matrix which is not positive semi-definite matrix
        weight  -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)
        epsilon -- the acceptable precision between near_psd and non_psd
        max_iter -- maximum iteration number

    Usage:
        Higham_psd_model=Higham_psd(non_psd,weight)
        psd=Higham_psd_model.psd
    '''
    # initialization
    def __init__(self,not_psd,weight,epsilon=1e-9,max_iter=1e10):
        self.__not_psd=not_psd
        self.__weight=weight
        self.run(epsilon=epsilon,max_iter=max_iter)
        self.F_compare_norm(weight)

    def Projection_U(self,A):
        # Projection to the Space U
        # we assume that the weight matrix is diagonal
        A_copy=A.copy()
        np.fill_diagonal(A_copy,1)
        return A_copy
        
    def Projection_S(self,A):
        # Projection to the Space S
        w_sqrt=np.sqrt(self.__weight)
        eig_val,eig_vec=np.linalg.eigh(w_sqrt @ A @ w_sqrt)
        eig_val[eig_val<0]=0
        A_plus=eig_vec @ np.diag(eig_val) @ eig_vec.T
        w_sqrt_inv=np.diag(1/np.diag(w_sqrt))
        ans = w_sqrt_inv @ A_plus @ w_sqrt_inv
        return ans
    
    def run(self,epsilon,max_iter):
        # iterating process
        Y=self.__not_psd
        F1=np.inf
        F2=self.calculate_F(Y,self.__weight)
        delta=0
        iteration=0
        neg_eig=0
        while abs(F1-F2)>epsilon or neg_eig>0:
            R=Y-delta
            X=self.Projection_S(R)
            delta=X-R
            Y=self.Projection_U(X)
            F1,F2=F2,self.calculate_F(Y,self.__weight)
            iteration+=1
            if iteration>max_iter:
                break
            eig_val=np.linalg.eigvals(Y)
            neg_eig=len(eig_val[eig_val<0])

        self.__F_norm=F2
        self.__psd=Y

    def F_compare_norm(self,weight):
        self.__F = self.compare_F(self.__psd,self.__not_psd,weight)
        
    @property
    def psd(self):
        return self.__psd 
    @property
    def F_norm(self):
        return self.__F_norm 
    
    @property
    def F(self):
        return self.__F


class Comparison(PSD,Non_psd_mat):
    '''
    It's used to compare the Rebonato and Jackel's Method, and Higham's Method.
    First, it confirms the results generated by two methods are PSD
    Second, it calculates the run time and precision of each method

    Parameters:
        size -- the size of Non-PSD matrix
    '''
    # initialization
    def __init__(self,size):

        self.__size=size
        self.__weight=np.eye(self.__size)
        self.__mat=self.non_psd_mat(self.__size)
        self.run()
        self.confirm()
    
    # confirms the results generated by two methods are PSD
    def confirm(self):
        mat=self.__mat
        weight=self.__weight
        RJ_psd=self.__RJ_psd.psd
        HG_psd=self.__HG_psd.psd
        print("Rebonato and Jackel's method of {0}*{0} matrix: ".format(self.__size))
        PSD.confirm(RJ_psd)
        print("Higham's method of {0}*{0} matrix: ".format(self.__size))
        PSD.confirm(HG_psd)
    
    # calculates the run time and precision of each method
    def run(self):
        t = time.time()
        self.__RJ_psd=near_psd(self.__mat,self.__weight)
        self.__RJ_run_time = time.time()-t

        t = time.time()
        self.__HG_psd=Higham_psd(self.__mat,self.__weight)
        self.__HG_run_time = time.time()-t

    @property
    def RJ_run_time(self):
        return self.__RJ_run_time

    @property
    def HG_run_time(self):
        return self.__HG_run_time   

    @property
    def HG_psd(self):
        return self.__HG_psd
        
    @property
    def RJ_psd(self):
        return self.__RJ_psd  
    
    @property
    def size(self):
        return self.__size 


def plot_summary(comparison):
    '''
    Use the Comparison class to calculate the nessary data and plot the result.
    '''
    n=len(comparison)
    method=["Rebonato & Jackel", "Higham"]
    fig, ax = plt.subplots(n,2,figsize=(14,6*n))

    for i in range(n):
        # plot the Run time  
        sns.barplot(x=method,y=[comparison[i].RJ_run_time,comparison[i].HG_run_time],ax=ax[i][0],palette=sns.color_palette("pastel"))
        ax[i][0].set_xlabel('Method')
        ax[i][0].set_ylabel('Run time (seconds)')
        ax[i][0].set_title("Run time of Matrix size={}".format(comparison[i].size),fontweight="bold")
        # plot the Weighted F norm
        sns.barplot(x=method,y=[comparison[i].RJ_psd.F,comparison[i].HG_psd.F],ax=ax[i][1],palette=sns.color_palette("pastel"))
        ax[i][1].set_xlabel('Method')
        ax[i][1].set_ylabel('Weighted F norm')
        ax[i][1].set_title("Weighted F norm of Matrix size={}".format(comparison[i].size),fontweight="bold")



class Cov_Generator:
    """
    Convariance Derivation through differnet combination of EW covariance, EW correlation, covariance and correlation.

    Parameter:
        data -- data which is used for get the EW covariance, EW correlation, covariance and correlation
    
    Usage:
        cov_generator=Cov_Generator(data)
        cov_generator.EW_cov_cov()
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


class Simulator(Weighted_F_norm):
    """
    Simulator for DirectSimulation & PCA_Simulation

    Parameter: 
        cov_mat -- covariance matrix
        draw_num -- the number of sample draw from simulation
    """

    def __init__(self,cov_mat,draw_num):
        self.__cov_mat=cov_mat
        self.__draw_num=draw_num
        self.__Direct_run_time=None
        self.__PCA_run_time=None
        self.__X=None

    def DirectSimulation(self):
        """Cholesky"""
        t = time.time()
        root=chol_psd(self.__cov_mat).root
        n=root.shape[0]
        rand_norm=np.random.normal(0, 1, size=(n, self.__draw_num))
        X= root @ rand_norm
        self.__Direct_run_time = time.time()-t
        self.__X=X
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
        self.__X=X
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