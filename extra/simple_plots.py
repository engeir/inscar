import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines

def chirp_sampling():
    order = [1, 2, 3, 5, 7, 10]
    t_max = 1e-3
    n = 1e4
    plt.figure()
    for o in order:
        # t = np.linspace(0, t_max, int(n))
        t = np.linspace(0, t_max**(1 / o), int(n))**o
        plt.plot(t, label=f'n = {o}')
    plt.ylabel('Sampled variable')
    plt.xlabel('Number of sample points')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    chirp_sampling()
