from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import primitive
from autograd.container_types import make_tuple

# This file is like kalman.py except we mark some things as primitives.

from util import T, get_n, unpack, bottom_right_indicator, sym, vs
from test import rand_natparam, dense_expectedstats, \
    rand_pair_potential, rand_node_potential, rand_psd, sample_backward
from kalman import logZ, expectedstats

### numerical util

def schur_complement(natparam):
  A, B, C = unpack(natparam)
  return C - np.matmul(T(B), np.linalg.solve(A, B))

def schur_complement_vjp(Dbar, natparam):
  n = get_n(natparam)
  A, B, C = unpack(natparam)
  X = np.linalg.solve(sym(A), B)
  I = np.reshape(np.eye(n+1), X.shape[:-2] + (n+1, n+1))
  XI = vs(( X, -I ))
  return np.matmul(XI, np.matmul(Dbar, T(XI)))

def schur_complement_vjp_vjp(Ebar, Dbar, natparam):
  n = get_n(natparam)
  A, B, C = unpack(natparam)
  Ainv = np.linalg.inv(sym(A))
  X = np.matmul(Ainv, B)
  I = np.reshape(np.eye(n+1), X.shape[:-2] + (n+1, n+1))
  XI = vs(( X, -I ))
  I0 = vs(( np.eye(n), np.zeros((n+1, n)) ))
  Xbar = np.matmul(np.matmul(Ebar[..., :n, :], XI), T(Dbar)) \
      + np.matmul(np.matmul(T(Ebar[..., :, :n]), XI), Dbar)
  g_natparam = sym(np.matmul(-I0, np.matmul(np.matmul(T(Ainv), Xbar), T(XI))))
  g_Dbar = np.matmul(T(XI), np.matmul(Ebar, XI))
  return g_Dbar, g_natparam

def logdet(A):
  return np.linalg.slogdet(A)[1]

def logdet_vjp(g, A):
  return g * np.linalg.inv(A)

def logdet_vjp_vjp(h, g, A):
  Ainv = np.linalg.inv(A)
  n = A.shape[-1]
  gg = np.sum(Ainv * h, (-1, -2))
  gA = -g * np.matmul(Ainv, np.matmul(h, Ainv))
  return gg, gA

### kalman util

def partial_marginalize(natparam):
  n = get_n(natparam)
  D = schur_complement(natparam)
  norm = 1./2 * logdet(-2*natparam[..., :n, :n]) + n/2. * np.log(2*np.pi)
  return D - bottom_right_indicator(n+1) * norm[..., None, None]

def partial_marginalize_vjp(Dbar, natparam):
  n = get_n(natparam)
  g1 = schur_complement_vjp(Dbar, natparam)
  g2 = -2*logdet_vjp(-1./2*Dbar[..., -1:, -1:], -2*natparam[..., :n, :n])
  I = np.eye(2*n+1)[:n]
  return g1 + np.matmul(I.T, np.matmul(g2, I))

def partial_marginalize_vjp_vjp(Ebar, Dbar, natparam):
  n = get_n(natparam)
  g_Dbar1, g_natparam1 = schur_complement_vjp_vjp(Ebar, Dbar, natparam)
  g_Dbar2, g_natparam2 = logdet_vjp_vjp(
      -2*Ebar[..., :n, :n], -1./2*Dbar[..., -1:, -1:], -2*natparam[..., :n, :n])
  g_Dbar = g_Dbar1 - 1./2 * g_Dbar2 * bottom_right_indicator(n+1)
  I = np.eye(2*n+1)[:n]
  g_natparam = g_natparam1 + -2*np.matmul(I.T, np.matmul(g_natparam2, I))
  return g_Dbar, g_natparam

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

### kalman filter, its vjp, and its vjp's vjp as primitives

def primitive_logZ(natparam):
  return kalman_filter(natparam)[0]

@primitive
def kalman_filter(natparam):
  n = get_n(natparam)
  filter_natparam = np.zeros_like(natparam)
  prediction_potential = np.zeros(natparam.shape[:-3] + (n+1, n+1))
  for t in xrange(natparam.shape[-3]):
    filter_natparam[..., t, :, :] = add_node_potential(
        prediction_potential, natparam[..., t, :, :])
    prediction_potential = partial_marginalize(filter_natparam[..., t, :, :])
  return np.sum(prediction_potential[..., -1, -1]), filter_natparam

@primitive
def kalman_filter_vjp(args):
  (g_logZ, g_filter_natparam), (_, filter_natparam) = args
  T, n = filter_natparam.shape[-3], get_n(filter_natparam)
  g_natparam = np.zeros_like(filter_natparam)
  g_prediction_potential = np.zeros(filter_natparam.shape[:-3] + (T+1, n+1, n+1))
  g_prediction_potential[..., -1, -1, -1] = g_logZ
  for t in xrange(g_natparam.shape[-3] - 1, -1, -1):
    g_prediction_potential[..., t, :, :], g_natparam[..., t, :, :] = \
        add_node_potential_vjp(
            g_filter_natparam[..., t, :, :] + partial_marginalize_vjp(
                g_prediction_potential[..., t+1, :, :], filter_natparam[..., t, :, :]))
  return g_natparam, g_prediction_potential
kalman_filter.defvjp(lambda g, ans, vs, gvs, natparam:
                     kalman_filter_vjp(make_tuple(g, ans))[0])

@primitive
def kalman_filter_vjp_vjp(g, ans, args):
  gg_natparam, gg_prediction_potential = g
  g_natparam, g_prediction_potential = ans
  _, (_, filter_natparam) = args
  gg_prediction_potential = np.copy(gg_prediction_potential)
  g_filter_natparam = np.zeros_like(filter_natparam)
  gg_filter_natparam = np.zeros_like(filter_natparam)
  for t in xrange(natparam.shape[-3]):
    gg_filter_natparam[..., t, :, :] = \
        add_node_potential(gg_prediction_potential[..., t, :, :], gg_natparam[..., t, :, :])
    out, g_filter_natparam[..., t, :, :] = partial_marginalize_vjp_vjp(
            gg_filter_natparam[..., t, :, :],
            g_prediction_potential[..., t+1, :, :], filter_natparam[..., t, :, :])
    gg_prediction_potential[..., t+1, :, :] += out
  gg_logZ = gg_prediction_potential[..., -1, -1, -1]

  g_args = (gg_logZ, gg_filter_natparam), (0., g_filter_natparam)
  return g_args
kalman_filter_vjp.defvjp(lambda g, ans, vs, gvs, args: kalman_filter_vjp_vjp(g, ans, args))

### sampling

def natural_sample(natparam, npr=npr.RandomState(0)):
    n = get_n(natparam)
    def helper(natparam):
        logZ, filter_natparam = kalman_filter(natparam)
        h = filter_natparam[..., :n, -1] + filter_natparam[..., -1, :n]
        L = T(np.linalg.inv(np.linalg.cholesky(-2.*filter_natparam[..., :n, :n])))
        eps = np.matmul(L, npr.normal(size=natparam.shape[:-2] + (n, 1)))
        return logZ + np.dot(np.ravel(h), np.ravel(eps))
    return grad(helper)(natparam)[..., :n, -1]


if __name__ == '__main__':
  # ### testing numerical util and vjp functions

  # ## schur complement
  # # setup
  # npr.seed(0)
  # n = 2
  # natparam = rand_pair_potential(n)

  # test1 = rand_node_potential(n)
  # test2 = rand_pair_potential(n)
  # to_scalar = lambda test: lambda fun: lambda *args: np.dot(np.ravel(fun(*args)), np.ravel(test))
  # scalar1 = to_scalar(test1)
  # scalar2 = to_scalar(test2)

  # # vjp
  # ans1 = grad(scalar1(schur_complement))(natparam)
  # ans2 = schur_complement_vjp(test1, natparam)
  # print np.allclose(ans1, ans2)

  # # hvp
  # ans1 = grad(scalar2(grad(scalar1(schur_complement))))(natparam)
  # _, ans2 = schur_complement_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 0 of vjp
  # ans1 = grad(scalar2(schur_complement_vjp), 0)(test1, natparam)
  # ans2, _ = schur_complement_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 1 of vjp
  # ans1 = grad(scalar2(schur_complement_vjp), 1)(test1, natparam)
  # _, ans2 = schur_complement_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # ## logdet
  # # setup
  # npr.seed(0)
  # n = 2
  # natparam = rand_psd(n)

  # scale = npr.randn()
  # test = rand_psd(n)
  # scalar = lambda fun: lambda *args: np.dot(np.ravel(fun(*args)), np.ravel(test))

  # # vjp
  # ans1 = grad(lambda x: scale * logdet(x))(natparam)
  # ans2 = logdet_vjp(scale, natparam)
  # print np.allclose(ans1, ans2)

  # # hvp
  # ans1 = grad(scalar(grad(lambda x: scale * logdet(x))))(natparam)
  # _, ans2 = logdet_vjp_vjp(test, scale, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 0 of vjp
  # ans1 = grad(scalar(logdet_vjp), 0)(scale, natparam)
  # ans2, _ = logdet_vjp_vjp(test, scale, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 1 of vjp
  # ans1 = grad(scalar(logdet_vjp), 1)(scale, natparam)
  # _, ans2 = logdet_vjp_vjp(test, scale, natparam)
  # print np.allclose(ans1, ans2)

  # ### testing kalman primitives
  # ## setup
  # npr.seed(0)
  # n = 2
  # natparam = rand_pair_potential(n)

  # test1 = rand_node_potential(n)
  # test2 = rand_pair_potential(n)
  # to_scalar = lambda test: lambda fun: lambda *args: np.dot(np.ravel(fun(*args)), np.ravel(test))
  # scalar1 = to_scalar(test1)
  # scalar2 = to_scalar(test2)

  # # val
  # ans1 = partial_marginalize(natparam)

  # # vjp
  # ans1 = grad(scalar1(partial_marginalize))(natparam)
  # ans2 = partial_marginalize_vjp(test1, natparam)
  # print np.allclose(ans1, ans2)

  # # hvp
  # ans1 = grad(scalar2(grad(scalar1(partial_marginalize))))(natparam)
  # _, ans2 = partial_marginalize_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 0 of vjp
  # ans1 = grad(scalar2(partial_marginalize_vjp), 0)(test1, natparam)
  # ans2, _ = partial_marginalize_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # # vjp wrt arg 1 of vjp
  # ans1 = grad(scalar2(partial_marginalize_vjp), 1)(test1, natparam)
  # _, ans2 = partial_marginalize_vjp_vjp(test2, test1, natparam)
  # print np.allclose(ans1, ans2)

  # ### testing kalman filter, its vjp, and its vjp's vjp
  # npr.seed(0)
  # n = 2
  # natparam = rand_natparam(3, n)

  # ans1 = logZ(natparam)
  # ans2 = primitive_logZ(natparam)
  # print np.allclose(ans1, ans2)

  # ans1 = grad(primitive_logZ)(natparam)
  # ans2 = dense_expectedstats(natparam)
  # print np.allclose(ans1, ans2)

  # ans1 = grad(lambda x: np.sum(np.sin(grad(primitive_logZ)(x))))(natparam)
  # ans2 = grad(lambda x: np.sum(np.sin(grad(logZ)(x))))(natparam)
  # print np.allclose(ans1, ans2)

  ### sampling
  npr.seed(0)
  n = 2
  natparam = rand_natparam(10, n)
  natparam[..., :-1, -1] = natparam[..., -1, :-1] = 0.  # zero mean

  ans1 = natural_sample(natparam, npr=npr.RandomState(0))
  ans2 = np.squeeze(sample_backward(kalman_filter(natparam)[1], npr=npr.RandomState(0)))
  print ans1
  print ans2
  print '\n'.join(map(str, np.isclose(ans1, ans2).all(1)))

# NOTES:
# - some of this code probably assumes incoming grads are symmetric
# - could save these from forward pass:
#     - A^{-1} (or L = chol(A) if we only want to do solves)
#     - A^{-1} B
#   basically compute all that stuff up front, then everything else is matmuls
#   and adds
