from __future__ import division
import numpy as np
import numpy.random as npr

from util import T, vs, hs, get_n, rand_psd, unpack
from kalman import expectedstats, node_to_pair

### testing util

def rand_node_potential(n):
  J = rand_psd(n)
  h = npr.randn(n, 1)
  const = npr.rand() + np.dot(h.T, np.linalg.solve(J, h))
  return -1./2 * vs(( hs(( J,    -h,    )),
                      hs(( -h.T, const, )), ))

def rand_pair_potential(n):
  return rand_node_potential(2*n)

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
  big_mu = np.matmul(big_Sigma, big_h[..., None])[..., 0]

  def stack_stats(t):
    start = t*n
    stop = t*n + 2*n
    Sigma = big_Sigma[..., start:stop, start:stop]
    mu = big_mu[..., start:stop][..., None]
    ExxT = Sigma + np.matmul(mu, T(mu))
    one = np.ones(ExxT.shape[:-2] + (1, 1))
    out = vs(( hs(( ExxT, mu,  )),
               hs(( T(mu), one, )), ))
    if out.shape[-1] == 2*n+1:
      return out
    return node_to_pair(out)

  return np.stack([stack_stats(t) for t in xrange(natparam.shape[-3])], axis=-3)

### script

if __name__ == '__main__':
  npr.seed(0)

  n = 2
  natparam = rand_natparam(2, n, (1, 1))

  ans1 = expectedstats(natparam)
  ans2 = dense_expectedstats(natparam)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
