import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines


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


if __name__ == '__main__':
    chirp_sampling()
