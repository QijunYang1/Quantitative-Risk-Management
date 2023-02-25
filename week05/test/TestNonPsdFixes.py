import unittest
from RiskPackage.NonPsdFixes import chol_psd,PSD,near_psd,Higham_psd
import pandas as pd
import numpy as np
class TestNonPsdFixes(unittest.TestCase):
    # Add ssert_frame_equal to the unittest

    def test_chol_psd_PD(self):
        # PD
        sigma=np.full([5,5],0.9)
        np.fill_diagonal(sigma, 1)
        root=chol_psd(sigma).root
        self.assertTrue(np.allclose(root@root.T,sigma))

    def test_chol_psd_PSD(self):
        # PSD
        sigma=np.full([5,5],0.9)
        np.fill_diagonal(sigma, 1)
        sigma[0][1]=1
        sigma[1][0]=1
        v,c=np.linalg.eig(sigma)
        root=chol_psd(sigma).root
        self.assertTrue(np.allclose(root@root.T,sigma))

    def test_near_psd(self):
        # Non-PSD
        non_psd=np.full([5,5],0.9)
        np.fill_diagonal(non_psd, 1)
        non_psd[0][1]=0.7357
        non_psd[1][0]=0.7357
        weight=np.eye(5)
        near_psd_model=near_psd(non_psd,weight)
        psd=near_psd_model.psd
        self.assertTrue(PSD.confirm(psd))

    def test_Higham_psd(self):
        # Non-PSD
        non_psd=np.full([5,5],0.9)
        np.fill_diagonal(non_psd, 1)
        non_psd[0][1]=0.7357
        non_psd[1][0]=0.7357
        weight=np.eye(5)
        Higham_psd_model=Higham_psd(non_psd,weight)
        psd=Higham_psd_model.psd
        self.assertTrue(PSD.confirm(psd))

unittest.main(argv=[''], verbosity=2, exit=False)