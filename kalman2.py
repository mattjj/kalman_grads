from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import primitive

# This file is like kalman.py except we mark some things as primitives.

from util import T, get_n, unpack, bottom_right_indicator, sym, vs
from test import rand_natparam, dense_expectedstats

### numerical util

@primitive
def schur_complement(natparam):
  A, B, C = unpack(natparam)
  return C - np.matmul(T(B), np.linalg.solve(A, B))

def schur_complement_vjp(Dbar, D, _vs, _gvs, natparam):
  n = get_n(natparam)
  A, B, C = unpack(natparam)
  X = np.linalg.solve(sym(A), B)
  _, I = np.broadcast_arrays(C, np.eye(n+1))
  XI = vs(( X, -I ))
  return np.matmul(XI, np.matmul(Dbar, T(XI)))

schur_complement.defvjp(schur_complement_vjp)

def logdet(natparam):
  A, _, _ = unpack(natparam)
  return np.linalg.slogdet(A)[1]

### kalman smoother primitives

def partial_marginalize(natparam):
  n = get_n(natparam)
  D = schur_complement(natparam)
  norm = 1./2 * logdet(-2*natparam) + n/2. * np.log(2*np.pi)
  return D - bottom_right_indicator(n+1) * norm[..., None, None]

def add_node_potential(node_potential, natparam):
  return natparam + node_to_pair(node_potential)

def node_to_pair(node_potential):
  n = node_potential.shape[-1] - 1
  I = np.delete(np.eye(2*n+1), slice(n, 2*n), axis=0)
  return np.matmul(I.T, np.matmul(node_potential, I))

### kalman smoother

def logZ(natparam):
  n = get_n(natparam)
  prediction_potential = np.zeros(natparam.shape[:-3] + (n+1, n+1))
  for t in xrange(natparam.shape[-3]):
    prediction_potential = partial_marginalize(
        add_node_potential(prediction_potential, natparam[..., t, :, :]))
  return np.sum(prediction_potential[..., -1, -1])

def expectedstats(natparam):
  return grad(logZ)(natparam)

if __name__ == '__main__':
  npr.seed(0)

  n = 2
  natparam = rand_natparam(2, n, (1, 1))

  ans1 = expectedstats(natparam)
  ans2 = dense_expectedstats(natparam)
  print np.allclose(ans1, ans2)
