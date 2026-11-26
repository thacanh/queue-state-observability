"""Test that changing an unfinished job's hidden demand cannot alter an estimate."""

import numpy as np

from routing_simulator import replay


bank = np.array([.5, 1., 3., 10.])
weights = (np.zeros((2, 16), np.float32), np.zeros(2, np.float32),
           np.zeros((2, 2), np.float32), np.zeros(2, np.float32),
           np.zeros((2, 2), np.float32), np.zeros(2, np.float32))
small = np.array([[0.,1.,4.,3.,5.], [.10,6.,12.,1.,8.]])
large = small.copy()
large[0,3] = 30.


def play(x, estimated):
    return replay(x, np.zeros(2,np.int64), x[:,1].copy(), np.array([1.5,1.]),
                  np.array([1,1]), np.ones(5), weights, 1, False, estimated, 1.,
                  bank, np.array([0,4]), np.r_[0.,np.cumsum(bank)], np.array([3.625]))


np.testing.assert_array_equal(play(small, True)[:,1:5], play(large, True)[:,1:5])
assert play(small, False)[1,1] != play(large, False)[1,1]
print("PASS: before either completion, both estimated trajectories route identically; oracle trajectories differ.")
