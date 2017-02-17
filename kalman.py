from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, primitive

### util

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

### numerical util

def schur_complement(natparam):
  A, B, C = unpack(natparam)
  return C - np.matmul(T(B), np.linalg.solve(A, B))

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

### testing

def rand_pair_potential(n):
  J11, J12, J22 = rand_psd(n), npr.randn(n,  n), rand_psd(n)
  h1, h2 = npr.randn(n, 1), npr.randn(n, 1)
  const = npr.randn(1, 1)
  return -1./2 * vs(( hs(( J11,    J12,   -h1,   )),
                      hs(( J12.T,  J22,   -h2,   )),
                      hs(( -h1.T,  -h2.T, const, )), ))

def rand_node_potential(n):
  J = rand_psd(n)
  h = npr.randn(n, 1)
  const = npr.randn(1, 1)
  return -1./2 * vs(( hs(( J,    -h,    )),
                      hs(( -h.T, const, )), ))

def rand_lds_natparam(T, n):
  return np.stack([rand_pair_potential(n) for _ in xrange(T-1)]
                  + [node_to_pair(rand_node_potential(n))])

def rand_natparam(T, n, leading_dims=()):
  sz = int(np.product(leading_dims))
  natparams = np.stack([rand_lds_natparam(T, n) for _ in xrange(sz)])
  return np.reshape(natparams, (leading_dims + (T, 2*n+1, 2*n+1)))

def make_dense(natparam):
  leading_dims, T, n = natparam.shape[:-3], natparam.shape[-3], get_n(natparam)
  big_J = np.zeros(leading_dims + (T*n, T*n))
  big_h = np.zeros(leading_dims + (T*n,))
  for t in xrange(T-1):
    start = t*n
    stop = t*n + 2*n
    big_J[..., start:stop, start:stop] += -2*natparam[..., t, :2*n, :2*n]
    big_h[..., start:stop] += 2*natparam[..., t, :2*n, -1]
  big_J[..., -n:, -n:] += -2*natparam[..., -1, :n, :n]
  big_h[..., -n:] += 2*natparam[..., -1, :n, -1]

  return big_J, big_h

def dense_expectedstats(natparam):
  n = get_n(natparam)
  big_J, big_h = make_dense(natparam)
  big_Sigma = np.linalg.inv(big_J)
  big_mu = np.dot(big_Sigma, big_h)

  def stack_stats(t):
    start = t*n
    stop = t*n + 2*n
    Sigma = big_Sigma[..., start:stop, start:stop]
    mu = big_mu[..., start:stop][..., None]
    ExxT = Sigma + np.matmul(mu, T(mu))
    out = vs(( hs(( ExxT,  mu,        )),
               hs(( T(mu), np.eye(1), )), ))
    if out.shape[-1] == 2*n+1:
      return out
    return node_to_pair(out)

  return np.stack([stack_stats(t) for t in xrange(natparam.shape[-3])])


if __name__ == '__main__':
  npr.seed(0)
  n = 2
  natparam = rand_natparam(2, n)

  print expectedstats(natparam)
  print dense_expectedstats(natparam)
