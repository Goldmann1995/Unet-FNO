import unittest
import numpy as np
from run_experiments import relative_error, bootstrap_mean, field_hash

class AnalysisTests(unittest.TestCase):
    def test_relative_error_uses_spatial_l1_in_percent(self):
        truth=np.array([[[2.,2.],[2.,2.]]])
        pred=np.array([[[3.,1.],[2.,2.]]])
        np.testing.assert_allclose(relative_error(pred,truth),[25.],rtol=1e-8)
    def test_bootstrap_resamples_realisations_not_steps(self):
        data=np.tile(np.array([1.,2.,3.]),(100,1))
        mean,lo,hi=bootstrap_mean(data,reps=1000)
        np.testing.assert_allclose(mean,[1,2,3])
        np.testing.assert_allclose(lo,mean)
        np.testing.assert_allclose(hi,mean)
    def test_hash_distinguishes_shape_and_content(self):
        a=np.arange(4,dtype=float).reshape(2,2)
        self.assertNotEqual(field_hash(a),field_hash(a.reshape(1,4)))
        self.assertNotEqual(field_hash(a),field_hash(a+1))

if __name__=='__main__':unittest.main()
