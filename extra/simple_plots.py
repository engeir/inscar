import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines
import scipy.constants as const
import scipy.special as sps


def chirp_sampling():
    order = [1, 2, 3, 5, 7, 10]
    style = ['-', '--', ':', '-.',
             (0, (3, 5, 1, 5, 1, 5)),
             (0, (3, 1, 1, 1, 1, 1))]
    t_max = 1e-3
    n = 1e4
    plt.figure(figsize=(8, 6))
    for o, _ in zip(order, style):
        t = np.linspace(0, t_max**(1 / o), int(n))**o
        plt.plot(t, 'k', label=f'n = {o}')  # , linestyle=s)
    plt.ylabel('Sampled variable')
    plt.xlabel('Number of sample points')
    labelLines(plt.gca().get_lines(), fontsize=9, zorder=2.5)
    # plt.savefig(f'../../report/master-thesis/figures/simpson_int_sampling.pdf',
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.tight_layout()
    plt.show()


def maxwell(x, T, m):
    f = (2 * np.pi * T * const.k / m)**(- 3 / 2) * \
        np.exp(- x**2 / (2 * T * const.k / m))
    return f


def kappa(x, T, m, k):
    theta2 = 2 * (k - 3 / 2) / k * T * const.k / m
    A = (np.pi * k * theta2)**(- 3 / 2) * \
        sps.gamma(k + 1) / sps.gamma(k - .5)
    f = A * (1 + x**2 / (k * theta2))**(- k - 1)
    return f


def vdf_plots():
    w = np.linspace(- 5e5, 5e5, 1e4)
    v = w * np.sqrt(const.electron_mass / (1000 * const.k))
    f = maxwell(w, 1000, const.electron_mass)
    norm = np.max(f)
    f /= norm
    style = [
        '-', '--', ':', '-.',
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1))
        ]
    plt.figure()
    plt.semilogy(v, f, 'k', label='Maxwellian', linestyle='-', linewidth=1.3)
    K = [2, 2.5, 3, 4, 10]
    for k, s in zip(K, style):
        f = kappa(w, 1000, const.electron_mass, k)
        f /= norm
        plt.semilogy(v, f, 'k', label=f'Kappa = {k}', linestyle=s, linewidth=.8)
    plt.legend()
    plt.ylim([1e-3, 1e1])
    plt.xlabel(r'$v/v_{th}$')
    plt.ylabel(r'$f/\max(f_{maxwellian})$')
    # plt.savefig(f'../../report/master-thesis/figures/vdf.pdf',
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


def chirp_z_fail():
    # This give a clear indication that the sampling is too low.
    # y1, y2, y3 represent the peaks of the electron, gyro and ion lines.
    # x = np.array([5e6, 1e7, 2e7, 3e7, 4e7, 5e7, 1e8])
    # y1 = np.array([2.078, 2.692, 3.322, 3.620, 3.7787, 3.8703, 3.8536])
    # y2 = np.array([.679, .6812, .6819, .6820, .6820, .6820, .6821])
    # y3 = np.array([6.288, 6.35, 6.384, 6.392, 6.390, 6.393, 6.403]) * 1e-4
    x = np.array([5e6, 1e7, 2e7, 4e7, 1e8])
    y1 = np.array([1.51449, 1.51601, 1.51639, 1.51649, 1.51651])
    y2 = np.array([.606318, .606359, .606369, .606372, .606373])
    y3 = np.array([9.42397, 9.43598, 9.438, 9.441, 9.442]) * 1e-4
    y1 -= np.min(y1)
    y2 -= np.min(y2)
    y3 -= np.min(y3)
    y1 /= np.max(y1)
    y2 /= np.max(y2)
    y3 /= np.max(y3)
    plot = plt.semilogx
    plt.figure()
    plot(x, y1, 'k', linestyle='-', label='Plasma line')
    plot(x, y2, 'k', linestyle=':', label='Gyro line')
    plot(x, y3, 'k', linestyle='--', label='Ion line')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$(f-f_\min)/(f_\max-f_\min)$')
    plt.legend()
    # plt.savefig(f'../../report/master-thesis/figures/chirp-z_artefact.pdf',
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    # chirp_sampling()
    # vdf_plots()
    chirp_z_fail()
