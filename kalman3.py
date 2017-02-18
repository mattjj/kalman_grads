from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import primitive

# This file is like kalman.py except we mark some things as primitives.

from util import T, get_n, unpack, bottom_right_indicator, sym, vs
from test import rand_natparam, dense_expectedstats

### numerical util

def schur_complement(natparam):
  A, B, C = unpack(natparam)
  return C - np.matmul(T(B), np.linalg.solve(A, B))

def schur_complement_vjp(Dbar, natparam):
  n = get_n(natparam)
  A, B, C = unpack(natparam)
  X = np.linalg.solve(sym(A), B)
  _, I = np.broadcast_arrays(C, np.eye(n+1))
  XI = vs(( X, -I ))
  return np.matmul(XI, np.matmul(Dbar, T(XI)))

def logdet(A):
  return np.linalg.slogdet(A)[1]

def logdet_vjp(g, A):
  Ainv = np.linalg.inv(A)
  n = A.shape[-1]
  I = np.eye(2*n+1)[:n]
  return g * np.matmul(I.T, np.matmul(Ainv, I))

### kalman smoother primitives

def partial_marginalize(natparam):
  n = get_n(natparam)
  D = schur_complement(natparam)
  norm = 1./2 * logdet(-2*natparam[..., :n, :n]) + n/2. * np.log(2*np.pi)
  return D - bottom_right_indicator(n+1) * norm[..., None, None]

def partial_marginalize_vjp(Dbar, natparam):
  return schur_complement_vjp(Dbar, natparam) \
      + -2*logdet_vjp(-1./2*Dbar[..., -1:, -1:], -2*natparam[..., :n, :n])

def add_node_potential(node_potential, natparam):
  return natparam + node_to_pair(node_potential)

def add_node_potential_vjp(G):
  return node_to_pair_vjp(G), G

def node_to_pair(node_potential):
  n = node_potential.shape[-1] - 1
  I = np.delete(np.eye(2*n+1), slice(n, 2*n), axis=0)
  return np.matmul(I.T, np.matmul(node_potential, I))

def node_to_pair_vjp(G):
  n = get_n(G)
  I = np.delete(np.eye(2*n+1), slice(n, 2*n), axis=0)
  return np.matmul(I, np.matmul(G, I.T))

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

return_first = lambda fun: lambda *args, **kwargs: fun(*args, **kwargs)[0]

@primitive
def primitive_logZ(natparam):
  n = get_n(natparam)
  filter_natparam = np.zeros_like(natparam)
  prediction_potential = np.zeros(natparam.shape[:-3] + (n+1, n+1))
  for t in xrange(natparam.shape[-3]):
    filter_natparam[..., t, :, :] = add_node_potential(
        prediction_potential, natparam[..., t, :, :])
    prediction_potential = partial_marginalize(filter_natparam[..., t, :, :])
  return np.sum(prediction_potential[..., -1, -1]), filter_natparam

def logZ_vjp(g, ans, vs, gvs, natparam):
  _, filter_natparam = ans
  g_logZ, g_filter_natparam = g
  n = get_n(natparam)
  G = g_logZ * bottom_right_indicator(n+1)
  g_natparam = np.zeros_like(natparam)
  for t in xrange(natparam.shape[-3] - 1, -1, -1):
    G, out = add_node_potential_vjp(partial_marginalize_vjp(G, filter_natparam[..., t, :, :]))
    g_natparam[..., t, :, :] += out
  return g_natparam

primitive_logZ.defvjp(logZ_vjp)
primitive_logZ = return_first(primitive_logZ)


if __name__ == '__main__':
  npr.seed(0)

  n = 2
  natparam = rand_natparam(3, n)

  ans1 = grad(primitive_logZ)(natparam)
  ans2 = dense_expectedstats(natparam)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
