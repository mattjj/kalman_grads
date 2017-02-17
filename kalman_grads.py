from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian, grad, hessian, primitive

# TODO this code assumes contracting against symmetric matrices

hs, vs = np.hstack, np.vstack
square = lambda X: np.dot(X, X.T)
rand_psd = lambda n: square(npr.randn(n, n))
rand_natparam = lambda n: vs(( hs(( rand_psd(n),      npr.randn(n, n),  npr.randn(n, 1) )),
                               hs(( np.zeros((n, n)), rand_psd(n),      npr.randn(n, 1) )),
                               hs(( np.zeros((1, n)), np.zeros((1, n)), npr.randn(1, 1) )), ))
unpack = lambda X: (X[:n, :n], X[:n, n:], X[n:, n:])
pack = lambda A, B, C: vs(( hs(( A,                  B, )),
                            hs(( np.zeros_like(B.T), C, )), ))
sym = lambda X: 0.5 * (X + X.T)

def bottom_right_indicator(n):
  arr = np.zeros((n, n))
  arr[-1, -1] = 1.
  return arr


def schur(natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, B, C = unpack(natparam)
  return C - np.dot(B.T, np.linalg.solve(A, B))

def grad_schur(natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, B, C = unpack(natparam)

  # NOTE: the solve is done in the schur forward pass
  X = np.linalg.solve(A, B)

  I = np.eye(n+1)
  XI = np.vstack((X, -I))

  def gradfun(Dbar):
    A, B, C = unpack(np.dot(XI, np.dot(Dbar, XI.T)))
    return pack(A, 2*B, C)

  return gradfun

def hess_schur(natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, B, C = unpack(natparam)

  # NOTE: the solve and matmul are done in schur forward pass
  X = np.linalg.solve(A, B)
  D = C - np.matmul(B.T, X)

  I = np.eye(n+1)
  XI = np.vstack((X, -I))

  def hessfun(Dbar, Ebar):
    Xbar = 2*np.dot(np.dot(Ebar[:n], XI), Dbar)
    ATinvXbar = np.linalg.solve(A.T, Xbar)
    return pack(-sym(np.dot(ATinvXbar, X.T)), ATinvXbar, np.zeros((n+1, n+1)))

  return hessfun

def logdet(natparam):
  A, _, _ = unpack(natparam)
  return np.linalg.slogdet(A)[1]

def hess_logdet(natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, _, _ = unpack(natparam)

  # NOTE: can use cholesky factor from forward computation
  Ainv = np.linalg.inv(A)

  def hessfun(X):
    Xsub, _, _ = unpack(X)
    ans = -np.dot(Ainv, np.dot(Xsub, Ainv))
    return pack(ans, np.zeros((n, n+1)), np.zeros((n+1, n+1)))

  return hessfun


def partial_marginalize(natparam):
  n = (natparam.shape[-1] - 1) // 2
  D = schur(natparam)
  return D + bottom_right_indicator(n+1) * logdet(natparam)

@primitive
def partial_marginalize_vjp(Dbar, D, vs, gvs, natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, _, _ = unpack(natparam)
  out = grad_schur(natparam)(Dbar)
  out[:n, :n] += Dbar[-1, -1] * np.linalg.inv(A)
  return out

@primitive
def partial_marginalize_hvp_natparam(Ebar, E, vs, gvs, Dbar, D, vs2, gvs2, natparam):
  n = (natparam.shape[-1] - 1) // 2
  A, _, _ = unpack(natparam)

  Ainv = np.linalg.inv(A)

  out = hess_schur(natparam)(Dbar, Ebar)
  Ebarsub, _, _ = unpack(Ebar)
  out[:n, :n] += -np.dot(Ainv, np.dot(Ebarsub, Ainv)) * Dbar[-1, -1]
  return out

@primitive
def partial_marginalize_hvp_D(Ebar, E, vs, gvs, Dbar, D, vs2, gvs2, natparam):
  # NOTE I don't think we need to implement this, I'm surprised that autograd is
  # calling it. We could track progenitors a little more carefully.
  # Do we ever need it?
  return np.zeros_like(Dbar)

primitive_partial_marginalize = primitive(partial_marginalize)
primitive_partial_marginalize.defvjp(partial_marginalize_vjp)
partial_marginalize_vjp.defvjp(partial_marginalize_hvp_natparam, argnum=4)
partial_marginalize_vjp.defvjp(partial_marginalize_hvp_D, argnum=1)


if __name__ == '__main__':
  npr.seed(0)

  n = 2
  natparam = rand_natparam(n)

  test1 = rand_psd(n+1)
  test2 = rand_psd(2*n+1)
  scalar = lambda fun: lambda x: np.trace(np.dot(test1.T, fun(x)))

  # test grad of schur
  ans1 = grad(scalar(schur))(natparam)
  ans2 = grad_schur(natparam)(test1)
  ans3 = np.einsum('ijkl,ij->kl', jacobian(schur)(natparam), test1)
  print ans1
  print ans2
  print ans3
  print np.allclose(ans1, ans2), np.allclose(ans1, ans3)
  print

  # test hess of schur
  ans1 = np.einsum('ijklmn,ij,kl->mn', hessian(schur)(natparam), test1, test2)
  ans2 = hess_schur(natparam)(test1, test2)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
  print

  # test hess of logdet
  ans1 = np.einsum('ijkl,ij->kl', hessian(logdet)(natparam), test2)
  ans2 = hess_logdet(natparam)(test2)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
  print

  # test partial_marginalize
  ans1 = grad(scalar(partial_marginalize))(natparam)
  ans2 = grad(scalar(primitive_partial_marginalize))(natparam)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
  print

  ans1 = np.einsum('ijkl,ij->kl', hessian(scalar(partial_marginalize))(natparam), test2)
  ans2 = np.einsum('ijkl,ij->kl', hessian(scalar(primitive_partial_marginalize))(natparam), test2)
  print ans1
  print ans2
  print np.allclose(ans1, ans2)
  print
