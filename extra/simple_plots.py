from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines
import scipy.constants as const
import scipy.special as sps

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'Ovo',
    # 'font.serif': 'Ovo',
    # 'mathtext.fontset': 'cm',
    # Use ASCII minus
    'axes.unicode_minus': False,
})

matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

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
        plt.plot(t, 'k', label=r'$n = {}$'.format(o))  # , linestyle=s)
    plt.ylabel('Sampled variable')
    plt.xlabel('Number of sample points')
    labelLines(plt.gca().get_lines(), fontsize=9, zorder=2.5)
    # plt.savefig(f'../../../report/master-thesis/figures/simpson_int_sampling.pgf', bbox_inches='tight')
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.tight_layout()
    plt.show()


def maxwell(x, T, m):
    f = (2 * np.pi * T * const.k / m)**(- 3 / 2) * \
        np.exp(- x**2 / (2 * T * const.k / m))
    return f


def f_0_gauss_shell(x, T, m):
    vth = np.sqrt(T * const.k / m)
    A = (2 * np.pi * T * const.k / m)**(- 3 / 2) / 2
    func = A * np.exp(- (np.sqrt(x**2) - 5 * vth)**2 / (2 * T * const.k / m)) + 10 * maxwell(x, T, m)
    return func / 11


def kappa(x, T, m, k):
    theta2 = 2 * (k - 3 / 2) / k * T * const.k / m
    A = (np.pi * k * theta2)**(- 3 / 2) * \
        sps.gamma(k + 1) / sps.gamma(k - .5)
    f = A * (1 + x**2 / (k * theta2))**(- k - 1)
    return f


def d_maxwell(x, T, m):
    f = (2 * np.pi * T * const.k / m)**(- 3 / 2) * (- x * m /
                                                    (T * const.k)) * np.exp(- x**2 / (2 * T * const.k / m))
    return f


def d_kappa(x, T, m, k):
    theta2 = 2 * (k - 3 / 2) / k * T * const.k / m
    A = (np.pi * k * theta2)**(- 3 / 2) * \
        sps.gamma(k + 1) / sps.gamma(k - .5)
    f = A * (- k - 1) * (2 * x / (k * theta2)) * \
        (1 + x**2 / (k * theta2))**(- k - 2)
    return f


def d_vdf_plots():
    # n = 5
    w = np.linspace(- 8e5, 8e5, int(1e4))
    v = w * np.sqrt(const.electron_mass / (1000 * const.k))
    f = d_maxwell(w, 1000, const.electron_mass)
    norm = np.max(f)
    f /= norm
    # f = f**(1 / n)
    f = abs(f)
    style = [
        '-', '--', ':', '-.',
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1))
    ]
    plot = plt.semilogy
    plt.figure()
    plot(v, f, 'k', label='Maxwellian', linestyle='-', linewidth=1.3)
    K = [2, 2.5, 3, 4, 10]
    for k, s in zip(K, style):
        f = d_kappa(w, 1000, const.electron_mass, k)
        f /= norm
        f = abs(f)
        # f = f**(1 / n)
        plot(v, f, 'k', label=r'$\kappa = {}$'.format(k), linestyle=s, linewidth=.8)
    plt.legend()
    plt.ylim([1e-5, 3e1])
    plt.xlabel(r'$v/v_{\mathrm{th}}$')
    plt.ylabel(r'$f_0/\max(f_{0,M})$')
    # plt.savefig(f'../../../report/master-thesis/figures/d_vdf.pgf', bbox_inches='tight')
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


def vdf_plots():
    T = 1000
    w_max = 1e6  # 13e5
    w = np.linspace(- w_max, w_max, int(1e4))
    v = w * np.sqrt(const.electron_mass / (T * const.k))
    f = maxwell(w, T, const.electron_mass)
    norm = np.max(f)
    f /= norm
    style = [
        '-', '--', ':', '-.',
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1))
    ]
    plot = plt.semilogy
    plt.figure()
    plot(v, f, 'k', label='Maxwellian', linestyle='-', linewidth=1.3)
    f = f_0_gauss_shell(w, T, const.electron_mass)
    f /= norm
    plot(v, f, 'k', label='Gauss shell', linestyle=':', linewidth=1.3)
    K = [2, 2.5, 3, 4, 10]
    for k, s in zip(K, style):
        f = kappa(w, T, const.electron_mass, k)
        f /= norm
        plot(v, f, 'k', label=r'$\kappa = {}$'.format(k), linestyle=s, linewidth=.8)
    plt.legend()
    plt.ylim([1e-5, 1e1])
    plt.xlabel(r'$v/v_{\mathrm{th}}$')
    plt.ylabel(r'$f_0/\max(f_{0,M})$')
    # plt.savefig(f'../../../report/master-thesis/figures/vdf.pgf', bbox_inches='tight')
                # bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


def chirp_z_fail():
    # This give a clear indication that the sampling is too low.
    # y1, y2, y3 represent the peaks of the electron, gyro and ion lines.
    x = np.array([5e6, 1e7, 2e7, 3e7, 4e7, 5e7, 1e8])
    y1 = np.array([2.078, 2.692, 3.322, 3.620, 3.7787, 3.8703, 3.8536])
    y2 = np.array([.679, .6812, .6819, .6820, .6820, .6820, .6821])
    y3 = np.array([6.288, 6.35, 6.384, 6.392, 6.390, 6.393, 6.403]) * 1e-4
    # x = np.array([5e6, 1e7, 2e7, 4e7, 1e8])
    # y1 = np.array([1.51449, 1.51601, 1.51639, 1.51649, 1.51651])
    # y2 = np.array([.606318, .606359, .606369, .606372, .606373])
    # y3 = np.array([9.42397, 9.43598, 9.438, 9.441, 9.442]) * 1e-4
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
    plt.ylabel(r'$(f-f_{\min})/(f_{\max}-f_{\min})$')
    plt.legend()
    # plt.savefig(f'../../../report/master-thesis/figures/chirp-z_artefact.pgf', bbox_inches='tight')
    #             bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


def l_Debye():
    k0 = 2 * 933e6 * 2 * np.pi / const.c  # Radar wavenumber
    T = np.linspace(1000, 10000, 1000)
    k = 3
    l_D = (const.epsilon_0 * const.k * T /
           (2e11 * const.elementary_charge**2))**.5
    l = l_D * ((k - 3 / 2) / (k - 1 / 2))**.5
    plt.figure()
    plt.plot(T, (k0 * l_D)**2)
    plt.plot(T, (k0 * l)**2)
    plt.legend(['l_D', 'l_D_k'])
    plt.show()


def twoD_gauss():
    xi, yi = np.linspace(- 1, 1, 100), np.linspace(- 1, 1, 100)
    x, y = np.meshgrid(xi, yi)
    e = np.exp(- (np.sqrt(x**2 + y**2) - .7)**2)
    plt.figure()
    plt.imshow(e, extent=[- 1, 1, - 1, 1])
    plt.show()


def sample_this(x, y, z):
    v = np.exp(- (np.sqrt(x**2 + y**2 + z**2) - 3)**2) * \
        (2 * np.pi**2)**(- 3 / 2) / 2
    if np.random.uniform() < 2 * v:
        return (x, y, z)
    return None


def gauss_3d():
    """Visual of a Gaussian shell distribution, both 2d and 3d.
    """
    xi = np.linspace(- 10, 10, 100)
    yi = np.linspace(- 10, 10, 100)
    zi = np.linspace(- 10, 10, 100)
    xs = []
    ys = []
    zs = []
    for x in xi:
        for y in yi:
            for z in zi:
                back = sample_this(x, y, 0)
                if back is not None:
                    xs.append(back[0])
                    ys.append(back[1])
                    zs.append(back[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs=0, zdir='z')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    # chirp_sampling()
    # vdf_plots()
    # d_vdf_plots()
    chirp_z_fail()
    # l_Debye()
    # twoD_gauss()
    # gauss_3d()
