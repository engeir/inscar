"""Main script for calculating the IS spectrum based on realistic parameters.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

# import config as cf
# import integrand_functions as intf
# import parallelization as para
import tool


# class ISR_Spectrum:
#     """Create ISR spectrum."""

#     def __init__(self, t_samples=1e2, f_samples=1e2):
#         self.t_samples = t_samples
#         self.f_samples = f_samples
#         self.Fe = 0
#         self.Fi = 0
#         self.setup()
#         self.p = para.InParallel(self.w, self.t_samples)

#     def setup(self):
#         self.f = np.linspace(- cf.F_MAX, cf.F_MAX, int(self.f_samples))
#         self.w = 2 * np.pi * self.f  # Angular frequency

#     def make_F(self, version):
#         """Calculate a ISR spectrum using the theory presented by Hagfors [1961].

#         Arguments:
#             version {str} -- decide which integral to use when calculating ISR spectrum

#         Raises:
#             SystemError: if the version is not valid / not found among the existing versions, an error is raised

#         Returns:
#             1D array -- two one dimensional numpy arrays for the frequency domain and the values of the spectrum
#         """
#         versions = ['hagfors', 'kappa', 'maxwell']
#         try:
#             if not version in versions:
#                 raise SystemError
#             else:
#                 print(f'Using version "{version}"', end='\r', flush=True)
#         except Exception:
#             tool.version_error(version, versions)
#         if version == 'hagfors':
#             func = intf.F_s_integrand
#         elif version == 'kappa':
#             func = intf.kappa_gordeyev
#         elif version == 'maxwell':
#             func = intf.maxwell_gordeyev
#         M_i = cf.MI * (const.m_p + const.m_n) / 2
#         w_c = tool.w_e_gyro(np.linalg.norm([cf.B], 2))
#         W_c = tool.w_ion_gyro(np.linalg.norm([cf.B], 2), M_i)
#         Lambda_e, Lambda_i = cf.NU_E / w_c, cf.NU_I / W_c
#         self.Fe = self.p.integrate(
#             w_c, const.m_e, cf.T_E, Lambda_e, cf.T_MAX_e, function=func)
#         self.Fi = self.p.integrate(
#             W_c, M_i, cf.T_I, Lambda_i, cf.T_MAX_i, function=func)

#     def h_spectrum(self, version, test=False):
#         if not isinstance(self.Fe, np.ndarray) or not isinstance(self.Fi, np.ndarray):
#             self.make_F(version)
#         w_c = tool.w_e_gyro(np.linalg.norm([cf.B], 2))
#         _, X = tool.make_X(w_c, const.m_e, cf.T_E)
#         X, F = tool.clip(X, 1e-4, 1e1, self.Fe, self.Fi)
#         Fe, Fi = F[0], F[1]
#         if test:
#             H = tool.H_func(X, 43, 300, Fe, Fi)
#             return X, H

#         kappa = [43, 172]
#         leg = []
#         plt.figure(figsize=(14, 8))
#         for c, k in enumerate(kappa):
#             plt.subplot(1, 2, c + 1)
#             for X_p in [300, 3., 1., .5, .1, .03]:
#                 H = tool.H_func(X, k, X_p, Fe, Fi)
#                 plt.loglog(X, H)
#                 if k == 43:
#                     leg.append(f'X_p = {X_p}')
#             plt.ylim([1e-3, 1e2])
#             plt.legend(leg, loc='lower left')
#             plt.title(f'Kappa = {k}')
#             plt.xlabel('f')
#             plt.ylabel('H(f)')
#             plt.grid(True, which="both", ls="-", alpha=0.3)
#         plt.show()

#     def isr_spectrum(self, version):
#         if not isinstance(self.Fe, np.ndarray) or not isinstance(self.Fi, np.ndarray):
#             self.make_F(version)
#         Xp = np.sqrt(1 / (2 * tool.L_Debye(cf.NE, cf.T_E)**2 * cf.K_RADAR**2))
#         f_scaled = self.f / 1e6
#         Is = cf.NE / (np.pi * self.w) * (np.imag(- self.Fe) * abs(1 + 2 * Xp**2 * self.Fi)**2 + (
#             4 * Xp**4 * np.imag(- self.Fi) * abs(self.Fe)**2)) / abs(1 + 2 * Xp**2 * (self.Fe + self.Fi))**2

#         return f_scaled, abs(Is)


def loglog(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.loglog(f, Is, 'r')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_y(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('10*log10(Power) [dB]')
    # plt.semilogy(f, Is, 'r')
    plt.plot(f, 10 * np.log10(Is), 'r')
    # plt.yscale('log')
    plt.minorticks_on()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def semilog_x(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    # plt.xlabel('log10(Frequency [MHz])')
    plt.ylabel('Power')
    plt.semilogx(f, Is, 'r')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()


def two_side_lin_plot(f, Is):
    plt.figure()
    plt.title('ISR spectrum')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power')
    plt.plot(f, Is, 'r')
    # plt.plot(- f, Is, 'r')
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.tight_layout()


def plot_IS_spectrum(version):
    f, Is = tool.isr_spectrum(version)
    save = input(
        'Press "y/yes" to save plot, any other key to dismiss.\t').lower()
    two_side_lin_plot(f, Is)
    loglog(f, Is)
    # semilog_x(f, Is)
    semilog_y(f, Is)
    if save in ['y', 'yes']:
        tt = time.localtime()
        the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}/{tt[4]}/{tt[5]}'
        os.makedirs('../../report/master-thesis/figures', exist_ok=True)
        plt.savefig(f'../../report/master-thesis/figures/{the_time}_hagfors.pdf',
                    bbox_inches='tight', format='pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    # TODO: when both functions are run using the same version, we do not need to calculate Fe and Fi twice.
    plot_IS_spectrum('hagfors')
    # tool.H_spectrum('kappa')
