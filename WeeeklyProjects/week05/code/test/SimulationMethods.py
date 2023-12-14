import unittest
from RiskPackage.SimulationMethods import PCA,Simulator
from sklearn import decomposition
import pandas as pd
import numpy as np

class TestSimulationMethods(unittest.TestCase):
    # Add ssert_frame_equal to the unittest

    def test_PCA(self):
        data = np.array([[1, 2, 3, 4],
                            [4, 9, 6, 8],
                            [7, 2, 9, 10]])
        cov = np.cov(data)
        PCA_model=PCA(cov)

        ans = decomposition.PCA()
        ans.fit(data.T)   # transposition

        res=PCA_model.percent_expalined()
        ans=ans.explained_variance_
        ans=ans.cumsum()/ans.sum()
        self.assertTrue(np.allclose(ans,res,atol=1e-7))
    
    def test_DirectSimulation(self):
        data = np.array([[1, 2, 3, 4],
                            [4, 9, 6, 8],
                            [7, 2, 9, 10]])
        ans = np.cov(data)
        simulator = Simulator(cov,100000)
        sample=simulator.DirectSimulation()
        res=np.cov(sample)
        self.assertTrue(np.allclose(ans,res,atol=1e-1))

    def test_PCASimulation(self):
        data = np.array([[1, 2, 3, 4],
                            [4, 9, 6, 8],
                            [7, 2, 9, 10]])
        ans = np.cov(data)
        simulator = Simulator(cov,100000)
        sample=simulator.PCA_Simulation(1)
        res=np.cov(sample)
        self.assertTrue(np.allclose(ans,res,atol=1e-1))
