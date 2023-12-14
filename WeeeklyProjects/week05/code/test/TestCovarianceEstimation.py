import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import unittest
from RiskPackage.CovarianceEstimation import EWMA
class TestCovarianceEstimation(unittest.TestCase):
    # Add ssert_frame_equal to the unittest
    def assertDataframeEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_EWMA_cov(self):
        df=pd.read_csv('DailyReturn.csv',index_col=0)
        data=np.array(df)
        model=EWMA(data,0.97) 
        res=pd.DataFrame(model.cov_mat)
        ans=pd.read_csv('test/Answer_EWMACov.csv',index_col=0)
        ans=pd.DataFrame(np.array(ans))
        self.assertEqual(res,ans)

    def test_EWMA_corr(self):
        df=pd.read_csv('DailyReturn.csv',index_col=0)
        data=np.array(df)
        model=EWMA(data,0.97) 
        res=pd.DataFrame(model.corr_mat)
        ans=pd.read_csv('test/Answer_EWMACorr.csv',index_col=0)
        ans=pd.DataFrame(np.array(ans))
        self.assertEqual(res,ans)