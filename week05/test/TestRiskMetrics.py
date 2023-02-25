import unittest
import pandas as pd
import numpy as np
from RiskPackage.RiskMetrics import RiskMetrics
from RiskPackage.ModelFitter import Norm,T
class TestRiskMetrics(unittest.TestCase):
    # Add ssert_frame_equal to the unittest

    def test_VaR(self):
        data=pd.read_csv('problem1.csv') # read the data
        data=np.array(data.values).reshape(data.size) # transform to array
        norm=Norm()
        norm.fit(data)
        fitted_norm=norm.fitted_dist
        res=RiskMetrics.VaR_dist(fitted_norm)
        ans=0.08125483146358596
        self.assertAlmostEqual(res,ans)
        
    def test_ES(self):
        data=pd.read_csv('problem1.csv') # read the data
        data=np.array(data.values).reshape(data.size) # transform to array
        norm=Norm()
        norm.fit(data)
        fitted_norm=norm.fitted_dist
        res=RiskMetrics.ES_dist(fitted_norm)
        ans=0.10167332455409121
        self.assertAlmostEqual(res,ans)
    
unittest.main(argv=[''], verbosity=2, exit=False)