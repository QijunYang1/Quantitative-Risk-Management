
from RiskPackage.CovarianceEstimation import EWMA
from RiskPackage.SimulationMethods import Simulator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from RiskPackage.ModelFitter import ModelFitter


'''
===============================================================================================================
Gaussian Copula (Simulate whatever distributions)
===============================================================================================================
'''

class GaussianCopula:
    ''' Construct the Gaussian Copula to simulate
    
        Parameters:
            dists(DataFrame) --- a group of distributions
            data(DataFrame) --- the data that fits the distributions will be used to generate the simulated sample

        Usage:
            copula=GaussianCopula(dists,data)
            sample=copula.simulate()
    '''
    def __init__(self,dists,data):
        self.models=dists
        self.data=data
    
    def simulate(self,NSim=5000):
        transform_data=pd.DataFrame()
        for name in self.data.columns:
            rt=np.array(self.data[name])
            # Use the CDF to transform the data to uniform universe
            # Use the standard normal quantile function to transform the uniform to normal 
            transform_data[name]=stats.norm.ppf(self.models[name][0].cdf(rt))
        # Spearman correlation
        corr_spearman = stats.spearmanr(transform_data,axis=0)[0]
        # Use PCA simulation
        simulator = Simulator(corr_spearman,NSim)
        # Simulate Normal & Transform to uniform
        SimU=stats.norm.cdf(simulator.PCA_Simulation(1),loc=0,scale=1)
        # Transform to Model Distribution
        simulatedResults = pd.DataFrame()
        for idx,name in enumerate(self.data.columns):
            simulatedResults[name] = self.models[name][0].ppf(SimU[idx,:])
        return simulatedResults.T

'''
===============================================================================================================
Fitted Fit Generalized T Distribution (helper function for VaR)
===============================================================================================================
'''

class T_fitter:
    '''T distribution MEL fitter'''
    def ll_t(self,parameter,x):
        # log likelihood 
        ll=np.sum(stats.t.logpdf(x=x,df=parameter[0],loc=0,scale=parameter[1])) # assume mean to be 0
        return -ll

    def MLE(self,x):
        cons=[ {'type':'ineq', 'fun':lambda x:x[1]} ] # standard deviation is non-negative
        parameter = np.array([x.size-1,1])
        MLE = minimize(self.ll_t, parameter, args = x, constraints = cons) # MLE
        return MLE.x


'''
===============================================================================================================
VaR for an Asset
===============================================================================================================
'''

class VaR:
    ''' Calculate the VaR of 1D array or dataframe of return due to specific distribution 
         1. normal
         2. normal+EWMA  
         3. T
         4. AR(1)
         5. Historical

        Parameters:
            data (Array/DataFrame) --- return matrix
            option (String) --- "Absolute":Absolute Value at risk; "Relative":Relative Value at risk
            alpha (Float) --- 1-alpha is confidence level
        
        Usage:
            VaR_normal=VaR(data).normal()
            VaR_normal_EWMA=VaR(data).normal('EWMA')
            VaR_T=VaR(data).T_dist()
            VaR_AR_1=VaR(data).AR_1()
            VaR_normal_historical=VaR(data).historical_simulation(0.05)
    '''
    def __init__(self,data,option="Absolute",alpha=0.05):
        # Absolute Value at risk or Relative Value at risk
        if option != "Absolute" and option != "Relative":
            raise Exception('Unknown option!')
        self.__option=option
        # 1-alpha is confidence level
        self.__alpha=alpha
        # returns data (DataFrame)
        self.__data=data

    def normal(self,option='Normal',plot=False):
        '''Assume returns follow normal distribution'''
        # assume mean to be 0
        mu=0
        # calculate the standard deviation
        if option == 'Normal': # use normal standard deviation
            std=self.__data.std() 
        elif option == 'EWMA': # use EW standard deviation
            model=EWMA(META,0.94) # assume lambda = 0.94
            std=np.sqrt(model.cov_mat)
        else: 
            raise Exception('Unknown option!')
         # calculate the VaR
        if self.__option=="Absolute":
            VaR=-stats.norm.ppf(self.__alpha,loc=mu,scale=std)
        else:
            VaR=self.__data.mean()-stats.norm.ppf(self.__alpha,loc=mu,scale=std)

        if plot:
            # plot the normal Probability density function
            max_val=self.__data.max()
            min_val=self.__data.min()
            x=np.linspace(min_val,max_val,1000)
            y=stats.norm.pdf(x=x,loc=mu,scale=std)
            plt.plot(x,y,color='brown')
            # fill VaR area
            plt.fill_between(x,y,where=x<-VaR,color="red", alpha=0.3)
            # plot the return data & its empirical kde
            sns.histplot(self.__data,kde=True,stat='density')
            # plot the VaR
            plt.axvline(-VaR,color='#FF6347')
            if option == 'Normal':
                plt.title("Normal")
                plt.legend(['Normal','VaR','historical'])
            else:
                plt.title("Normal with EW standard deviation")
                plt.legend(['Normal','VaR','historical'])
        return VaR
        
    def T_dist(self,plot=False):
        para=T_fitter().MLE(self.__data)
        mu=0 # assume mean to be 0
        df=para[0] # degree of freedom of T distribution
        std=para[1] # standard deviation
        # calculate the VaR
        if self.__option=="Absolute":
            VaR=-stats.t.ppf(self.__alpha,df=df,loc=mu,scale=std)
        else:
            VaR=self.__data.mean()-stats.t.ppf(self.__alpha,df=df,loc=mu,scale=std)
        if plot:
            # plot the T Probability density function
            max_val=self.__data.max()
            min_val=self.__data.min()
            x=np.linspace(min_val,max_val,1000)
            y=stats.t.pdf(x=x,df=df,loc=mu,scale=std)
            plt.plot(x,y,color='brown')
            # fill VaR area
            plt.fill_between(x,y,where=x<-VaR,color="red", alpha=0.3)
            # plot the return data & its empirical kde
            sns.histplot(self.__data,kde=True,stat='density')
            # plot the VaR
            plt.axvline(-VaR,color='#FF6347')
            plt.title("MLE fitted T distribution")
            plt.legend(['T distribution','VaR','historical'])
        return VaR

    def AR_1(self,plot=False):
        # Use AR(1) fitter to find the best cofficience
        mod = sm.tsa.arima.ARIMA(META.values, order=(1, 0, 0))
        res = mod.fit()
        const=0 # constant number
        ar_L1=res.params[1] # cofficience of Lag 1
        sigma2=res.params[2] # variance of error term

        # AR(1) is also normal
        mu=0
        std=np.sqrt(sigma2/(1-ar_L1))

        # calculate the VaR
        if self.__option=="Absolute":
            VaR=-stats.norm.ppf(self.__alpha,loc=mu,scale=std)
        else:
            VaR=self.__data.mean()-stats.norm.ppf(self.__alpha,loc=mu,scale=std)
        
        if plot:
            # plot the AR(1) Probability density function
            max_val=self.__data.max()
            min_val=self.__data.min()
            x=np.linspace(min_val,max_val,1000)
            y=stats.norm.pdf(x=x,loc=mu,scale=std)
            plt.plot(x,y,color='brown')
            # fill VaR area
            plt.fill_between(x,y,where=x<-VaR,color="red", alpha=0.3)
            # plot the return data & its empirical kde
            sns.histplot(self.__data,kde=True,stat='density')
            # plot the VaR
            plt.axvline(-VaR,color='#FF6347')
            plt.title("fitted AR(1)")
            plt.legend(['fitted AR(1)','VaR','historical'])
        return VaR
    
    def historical_simulation(self,alpha,plot=False):
        size=self.__data.shape[0]
        rt=self.__data.sample(n=size,replace=True)
        # calculate the VaR
        if self.__option=="Absolute":
            VaR=-np.quantile(rt,alpha)
        else:
            VaR=rt.mean()-np.quantile(rt,alpha)
        if plot:
            # plot the historical kde
            sns.kdeplot(rt,color='brown')

            # plot the return data & its empirical kde
            ax = sns.histplot(self.__data,kde=True,stat='density')
            
            # fill VaR area
            # Get the two lines from the axes to generate shading
            l = ax.lines[0]
            # Get the xy data from the lines so that we can shade
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x,y,where=x<-VaR,color="red", alpha=0.3)

            # plot the VaR
            plt.axvline(-VaR,color='#FF6347')
            plt.title("historical simulation")
            plt.legend(['historical simulation','VaR','historical'])
        return VaR
        


'''
===============================================================================================================
VaR for Portfolio
===============================================================================================================
'''

class VaR_portfolio:
    '''
    Differnet Method to get the VaR of stock portfolio
    
    1. delta_normal
    2. normal_MC
    3. historical_simulation

    The fromat of portfolio, returns, price should be same as the file in the current directory.

    Usage:
        All=VaR_portfolio(portfolio,rt,price).delta_normal(0.05)
        All=VaR_portfolio(portfolio,rt,price).normal_MC(plot=True,p_name='All')
        All=VaR_portfolio(portfolio,rt,price).historical_simulation(plot=True,p_name='All')

    '''
    # initialization
    def __init__(self,portfolio,returns,price):
        '''The fromat of data should be same as the file in the current directory.'''
        # information about portfolio
        self.__portfolio=portfolio
        # returns information about companies
        self.__returns=returns
        # price about stock
        self.__price=price

    def delta_normal(self,alpha=0.05,option='EWMA'):
        ''' Delta Normal method to calculate the VaR of the Stock portfolio
            
            Parmeter:
                option='EWMA' or 'std'
        '''
        stocks = self.__portfolio.Stock.values # stocks of portfilio
        rt=self.__returns[stocks] # return of each stock of the portfilio
        portfolio=self.__portfolio.set_index('Stock') # set the portfolio index to be Stock
        holding=portfolio.loc[stocks].Holding # get holding of each stock of the portfilio
        current_price=self.__price[stocks].iloc[-1,:] # get current price of each stock of the portfilio
        current_position = holding * current_price # current position of each stock of the portfilio
        PV = current_position.sum() # portfolio value
        delta=current_position/PV # calculate the delta
        stocks = self.__portfolio.Stock.values
        rt=self.__returns[stocks]

        if option=='EWMA':
            # Calculate EWMA covariance
            model=EWMA(rt.values,0.94) # assume lambda = 0.94
            cov_mat=model.cov_mat
        elif option=='std':
            # Calculate standard deviation
            cov_mat=rt.cov()

        # calculate the std of portfolio
        std = np.sqrt(delta @ cov_mat @ delta)
        # get the VaR of the portfolio
        VaR_p = PV*(-stats.norm.ppf(alpha))*std
        return VaR_p

    def normal_MC(self,alpha=0.05,option='EWMA',method='PCA',draw_num=100000,pct=1,plot=False,VaROption='Absolute',p_name=''):
        ''' Use Monte Carlo Methods to simulate the price of each stock of portfolio
            then calculate the value of the portfolio
            
            Parmeter:
                option: 'EWMA' or 'std'
                method: 'PCA' or 'Cholesky'
                VaROption: 'Absolute' VaR or 'Relative' VaR
                draw_num: number of path simulated
                pct: the percentage of variance expained by PCA
                plot: plot the simulated path or not
                alpha: 1-alpha is confidence level
                p_name: portfolio name
        '''
        stocks = self.__portfolio.Stock.values # stocks of portfilio
        rt=self.__returns[stocks] # return of each stock of the portfilio
        portfolio=self.__portfolio.set_index('Stock') # set the portfolio index to be Stock
        holding=portfolio.loc[stocks].Holding # get holding of each stock of the portfilio
        current_price=self.__price[stocks].iloc[-1,:] # get current price of each stock of the portfilio

        # get the covariance
        if option=='EWMA':
            # Calculate EWMA covariance
            model=EWMA(rt.values,0.94) # assume lambda = 0.94
            cov_mat=model.cov_mat
        elif option=='std':
            # Calculate standard deviation
            cov_mat=rt.cov()
        else:
            raise Exception("Unknown option!")
        
        # MC to get the simulated return
        simulator = Simulator(cov_mat,draw_num)
        if method=='PCA':
            simulated_rt=simulator.PCA_Simulation(pct)
        elif method=='Cholesky':
            simulated_rt=simulator.DirectSimulation()
        else:
            raise Exception("Unknown method!")
        # simulated price
        simulate_price = np.expand_dims(current_price,1).repeat(draw_num,axis=1) * simulated_rt
        # simulated position
        simulate_position=np.expand_dims(holding,1).repeat(draw_num,axis=1) * simulate_price
        # simulated portfolio value
        simulate_PV=pd.DataFrame(simulate_position).sum()
        # sort
        simulate_PV=pd.DataFrame(simulate_PV.sort_values(ascending=True))
        self.simulate_PV=simulate_PV

        if VaROption=="Absolute":
            VaR_p=-np.quantile(simulate_PV,alpha)
        elif VaROption=='Relative':
            VaR_p=simulate_PV.mean()-np.quantile(simulate_PV,alpha)
        else:
            raise Exception("Unknown VaROption!")
        
        
        if plot:
            # add current Portfolio value
            simulate_PV[1]=simulate_PV
            simulate_PV[0]=0
            plot_data=simulate_PV.T
            # plot
            fig, ax = plt.subplots(1,2,figsize=(14,6))
            plot_data.plot(ax=ax[0],legend=False,xlabel='Time',ylabel='Price',title="Mote Carlo Simulation({} path) for portfolio {}".format(draw_num,p_name))
            sns.histplot(data=simulate_PV[1],kde=True,stat="density",ax=ax[1])

            # fill VaR area
            # Get the two lines from the axes to generate shading
            l = ax[1].lines[0]
            # Get the xy data from the lines so that we can shade
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax[1].fill_between(x,y,where=x<-VaR_p,color="red", alpha=0.3)
            
            # plot the VaR
            plt.axvline(-VaR_p,color='#FF6347')
            plt.title("Monte Carlo Simulated VaR($) of Portfolio {}".format(p_name))
            plt.legend(['MC simulation kde','VaR'])
        return VaR_p




    def historical_simulation(self,alpha=0.05,draw_num=10000,plot=False,p_name='',VaROption='Absolute'):
        ''' Use historical returns as dataset, draw sample from it to simulate the 
            potential loss (VaR)

            Parameter:
                draw_num: number of path simulated
                p_name: portfolio name
                VaROption: 'Absolute' VaR or 'Relative' VaR
        '''
        stocks = self.__portfolio.Stock.values # stocks of portfilio
        rt=self.__returns[stocks] # return of each stock of the portfilio
        portfolio=self.__portfolio.set_index('Stock') # set the portfolio index to be Stock
        holding=portfolio.loc[stocks].Holding # get holding of each stock of the portfilio
        current_price=self.__price[stocks].iloc[-1,:] # get current price of each stock of the portfilio

        # sampling from the historical returns
        size=draw_num
        historical_rt=rt.sample(n=size,replace=True)

        # simulated price
        simulate_price = np.expand_dims(current_price,1).repeat(draw_num,axis=1) * (historical_rt.T)
        # simulated position
        simulate_position=np.expand_dims(holding,1).repeat(draw_num,axis=1) * simulate_price
        # simulated portfolio value
        simulate_PV=pd.DataFrame(simulate_position).sum()
        # sort
        simulate_PV=pd.DataFrame(simulate_PV.sort_values(ascending=True))
        self.simulate_PV=simulate_PV

        if VaROption=="Absolute":
            VaR_p=-np.quantile(simulate_PV,alpha)
        elif VaROption=='Relative':
            VaR_p=simulate_PV.mean()-np.quantile(simulate_PV,alpha)
        else:
            raise Exception("Unknown VaROption!")

        if plot:
            # add a column of current Portfolio value
            simulate_PV[1]=simulate_PV
            simulate_PV[0]=0
            plot_data=simulate_PV.T
            # plot
            fig, ax = plt.subplots(1,2,figsize=(14,6))
            plot_data.plot(ax=ax[0],legend=False,xlabel='Time',ylabel='Price',title="Historical Simulation({} path) for portfolio {}".format(draw_num,p_name))
            sns.histplot(data=simulate_PV[1],kde=True,stat="density",ax=ax[1])

            # fill VaR area
            # Get the two lines from the axes to generate shading
            l = ax[1].lines[0]
            # Get the xy data from the lines so that we can shade
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax[1].fill_between(x,y,where=x<-VaR_p,color="red", alpha=0.3)

            # plot the VaR
            plt.axvline( x=-VaR_p ,color='#FF6347')
            plt.title("Historical Simulated VaR($) of Portfolio {}".format(p_name))
            plt.legend(['Historical simulation kde','VaR'])

        return VaR_p


    def Copula_MC(self,dist,alpha=0.05,draw_num=10000,pct=1,plot=False,VaROption='Absolute',p_name=''):
        ''' Use Copula Monte Carlo Methods(PCA) to simulate the price of each stock of portfolio
            then calculate the value of the portfolio
            
            Parmeter:
                VaROption: 'Absolute' VaR or 'Relative' VaR
                draw_num: number of path simulated
                plot: plot the simulated path or not
                alpha: 1-alpha is confidence level
                p_name: portfolio name
        '''
        stocks = self.__portfolio.Stock.values # stocks of portfilio
        rt=self.__returns[stocks] # return of each stock of the portfilio
        portfolio=self.__portfolio.set_index('Stock') # set the portfolio index to be Stock
        holding=portfolio.loc[stocks].Holding # get holding of each stock of the portfilio
        current_price=self.__price[stocks].iloc[-1,:] # get current price of each stock of the portfilio

        # Fit the data with Model to get a group of distributions
        dists=ModelFitter(dist).fit(rt)
        # Construct Copula
        copula=GaussianCopula(dists,rt)
        # Simulate
        simulated_rt=copula.simulate(NSim=draw_num)
        # simulated price
        simulate_price = np.expand_dims(current_price,1).repeat(draw_num,axis=1) * simulated_rt
        # simulated position
        simulate_position=np.expand_dims(holding,1).repeat(draw_num,axis=1) * simulate_price
        # simulated portfolio value
        simulate_PV=pd.DataFrame(simulate_position).sum()
        # sort
        simulate_PV=pd.DataFrame(simulate_PV.sort_values(ascending=True))
        self.simulate_PV=simulate_PV

        if VaROption=="Absolute":
            VaR_p=-np.quantile(simulate_PV,alpha)
        elif VaROption=='Relative':
            VaR_p=simulate_PV.mean()-np.quantile(simulate_PV,alpha)
        else:
            raise Exception("Unknown VaROption!")
        
        
        if plot:
            # add current Portfolio value
            simulate_PV[1]=simulate_PV
            simulate_PV[0]=0
            plot_data=simulate_PV.T
            # plot
            fig, ax = plt.subplots(1,2,figsize=(14,6))
            plot_data.plot(ax=ax[0],legend=False,xlabel='Time',ylabel='Price',title="Copula Mote Carlo Simulation({} path) for portfolio {}".format(draw_num,p_name))
            sns.histplot(data=simulate_PV[1],kde=True,stat="density",ax=ax[1])

            # fill VaR area
            # Get the two lines from the axes to generate shading
            l = ax[1].lines[0]
            # Get the xy data from the lines so that we can shade
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax[1].fill_between(x,y,where=x<-VaR_p,color="red", alpha=0.3)
            
            # plot the VaR
            plt.axvline(-VaR_p,color='#FF6347',linestyle='--')
            plt.title("Copula Monte Carlo Simulated VaR($) of Portfolio {}".format(p_name))
            plt.legend(['Copula MC simulation kde','VaR'])
        
        return VaR_p


'''
===============================================================================================================
VaR & ES
===============================================================================================================
'''

class RiskMetrics:
    @staticmethod
    def VaR_dist(dist,alpha=0.05):
        '''Given a distribution and alpha, calculate the corresponding VaR'''
        return -dist.ppf(alpha)

    @staticmethod
    def ES_dist(dist,alpha=0.05):
        '''Given a distribution and alpha, calculate the corresponding Expected Shortfall'''
        lb=-np.inf   
        ub=dist.ppf(alpha)
        return -dist.expect(lb=lb,ub=ub)/alpha # integral
    
    @staticmethod
    def VaR_historical(data,alpha=0.05):
        '''Given a dataset(array), calculate the its historical VaR'''
        data.sort()
        n=round(data.shape[0]*alpha)
        return -data[n-1]
    
    @staticmethod
    def ES_historical(data,alpha=0.05):
        '''Given a dataset(array), calculate the its historical Expected Shortfall'''
        data.sort()
        n=round(data.shape[0]*alpha)
        return -data[:n].mean()

    