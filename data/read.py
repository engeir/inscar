import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.constants as const


def f_0_maxwell(E, T):
    # E given in eV
    E = E.copy()
    E *= const.eV  # scale to unit Joule
    v = (2 * E / const.m_e)**.5
    v_th = T * const.k / const.m_e
    # print(np.max(v) / v_th)
    # NOTE: Normalized to 1D
    A = (2 * np.pi * T * const.k / const.m_e)**(- 1 / 2)
    func = A * np.exp(- v**2 / (2 * T * const.k / const.m_e))
    return func


energy = np.linspace(0, 600, 1000)  # eV
# assumes thermal temperature of 1000 K
e_T = 1000
f_0 = f_0_maxwell(energy, e_T)

x = loadmat('Arecibo-photo-electrons/fe_zmuE-01.mat')
data = x['fe_zmuE']
dim = data.shape

avg_over_pitch = np.einsum('ijk->ik', data)  # removes j-dimansion through dot-product
count = np.argmax(avg_over_pitch, 0)
idx = np.argmax(np.bincount(count))
f_1 = avg_over_pitch[idx, :]
energies = np.linspace(1, f_1.shape[0], f_1.shape[0])
new_f1 = np.interp(energy, energies, f_1, left=0)
f0_f1 = f_0 + new_f1

plt.figure()
# plt.semilogy(energy, f_0, '--')
# plt.semilogy(energies, f_1, 'r:')
# plt.semilogy(energy, new_f1, 'k-.')
plt.semilogy(energy, f0_f1, '-')
# plt.legend(['f_0', 'f_1', 'new_f1', 'sum'])
# plt.ylim([1e-19, 1e-5])
plt.show()
