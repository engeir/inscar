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
    for o, s in zip(order, style):
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


if __name__ == '__main__':
    # chirp_sampling()
    vdf_plots()
