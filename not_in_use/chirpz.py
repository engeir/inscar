"""Chirp z-Transform.

As described in

Rabiner, L.R., R.W. Schafer and C.M. Rader.
The Chirp z-Transform Algorithm.
IEEE Transactions on Audio and Electroacoustics, AU-17(2):86--92, 1969
https://ieeexplore.ieee.org/abstract/document/1162034
https://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
https://gist.github.com/endolith/2783807
"""

import numpy as np


def chirpz(x, A, W, M):
    """Compute the chirp z-transform.

    The discrete z-transform,

    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}

    is calculated at M points,

    z_k = AW^-k, k = 0,1,...,M-1

    for A and W complex, which gives

    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}

    """
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex, x.dtype) or np.issubdtype(np.float, x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x, dtype=np.complex)

    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))

    n = np.arange(N, dtype=float)
    y = np.power(A, -n) * np.power(W, n**2 / 2.) * x
    Y = np.fft.fft(y, L)

    v = np.zeros(L, dtype=np.complex)
    v[:M] = np.power(W, -n[:M]**2/2.)
    v[L-N+1:] = np.power(W, -n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)

    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W, k**2 / 2.)

    return g
