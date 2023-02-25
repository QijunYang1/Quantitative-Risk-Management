import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

'''
===============================================================================================================
Cholesky Factorization For Positive Semi-definite Matrix
===============================================================================================================
'''

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
        matrix(Array)  --  Sysmetric Positive Definite (or Positive Semi-definite) 
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

'''
===============================================================================================================
Weighted Frobenius Norm (Assume Diagonal)
===============================================================================================================
'''

class Weighted_F_norm:
    '''
    Given the weight matrix(Array), calculate the Weighted Frobenius Norm. (Assume it's diagonal)
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


'''
===============================================================================================================
Positive Semi-definite Matrix Comfirmation
===============================================================================================================
'''

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
    PSD class is used for Positive Semi-Definite Matrix Confirmation.
    psd(Array) -- matrix to be confirmed
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
            return True
        else:
            raise NegativeEigError("Matrix has negative eigenvalue.")

'''
===============================================================================================================
Rebonato and Jackel's Method (Non-PSD Correlation Matrix Fix)
===============================================================================================================
'''

class near_psd(Weighted_F_norm):
    '''
    Rebonato and Jackel's Method to get acceptable PSD matrix 
    
    Parameters:
        not_psd (Array) -- the matrix which is not positive semi-definite matrix
        weight (Array) -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)

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

'''
===============================================================================================================
Higham's Method to Find Nearest PSD Correlation Matrix (Non-PSD Correlation Matrix Fix)
===============================================================================================================
'''

class Higham_psd(Weighted_F_norm,chol_psd):
    '''
    Higham's Method to get nearest PSD matrix under the measure of Weighted Frobenius Norm
    
    Parameters:
        not_psd (Array) -- the matrix which is not positive semi-definite matrix
        weight (Array) -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)
        epsilon (Float)-- the acceptable precision between near_psd and non_psd
        max_iter (Integer)-- maximum iteration number

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