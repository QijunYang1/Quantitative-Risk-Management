import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import unittest
from RiskPackage.CalculateReturn import return_calculate
class TestCalculateReturn(unittest.TestCase):
    # Add ssert_frame_equal to the unittest
    def assertDataframeEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_return_calculate(self):
        df=pd.read_csv('DailyPrices.csv',index_col='Date')
        # calculate the returns
        rt=return_calculate(df)
        res=return_calculate(rt)
        ans=pd.read_csv('test/Answer_return_calculate.csv',index_col='Date')
        
        self.assertEqual(res,ans)