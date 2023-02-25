
import pandas as pd
import numpy as np
import unittest
from RiskPackage.ModelFitter import Norm,T
class TestModelFitter(unittest.TestCase):
    # Add ssert_frame_equal to the unittest

    def test_Norm(self):
        data=pd.read_csv('problem1.csv') # read the data
        data=np.array(data.values).reshape(data.size) # transform to array
        # Fit the normal distribution
        norm=Norm()
        norm.fit(data)
        res=norm.fitted_parameters
        ans=np.array([-0.00087983,  0.04886453])
        self.assertTrue(np.allclose(ans,res,atol=1e-7))

    def test_T(self):
        data=pd.read_csv('problem1.csv') # read the data
        data=np.array(data.values).reshape(data.size) # transform to array
        t=T()
        t.fit(data)
        ans=t.fitted_parameters
        res=np.array([ 4.25128435e+00, -9.40174459e-05,  3.64400151e-02])
        self.assertTrue(np.allclose(ans,res,atol=1e-7))

unittest.main(argv=[''], verbosity=2, exit=False)