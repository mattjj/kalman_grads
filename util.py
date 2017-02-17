from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr

T = lambda X: np.swapaxes(X, -1, -2)
get_n = lambda natparam: (natparam.shape[-1] - 1) // 2
_unpack = lambda n, X: (X[..., :n, :n], 1./2*(X[..., :n, n:] + T(X[..., n:, :n])), X[..., n:, n:])
unpack = lambda X: _unpack(get_n(X), X)

hs, vs = np.hstack, np.vstack
square = lambda X: np.dot(X, X.T)
rand_psd = lambda n: square(npr.randn(n, n))

def bottom_right_indicator(n):
  arr = np.zeros((n, n))
  arr[-1, -1] = 1.
  return arr

