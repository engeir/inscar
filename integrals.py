import numpy as np


def Fe_integrand(y, X, Xe, Lambda_e, theta):
    """Calculate the integral in the expression for F_e in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    acf = np.exp(- Lambda_e * y - (1 / (2 * Xe**2)) * (np.sin(theta)**2 *
                                                       (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
    imag_part = np.exp(- 1j * (X / Xe) * y)
    W = np.exp(- 1j * (X / Xe) * y - Lambda_e * y - (1 / (2 * Xe**2)) *
               (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
    return acf  # , acf, imag_part


def Fi_integrand(y, X, Xi, Lambda_i, theta):
    """Calculate the integral in the expression for F_i in Hagfors.

    Arguments:
        y {float} -- integration variable

    Returns:
        float -- the value of the integral
    """
    acf = np.exp(- Lambda_i * y - (1 / (2 * Xi**2)) * (np.sin(theta)
                                                       ** 2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
    imag_part = np.exp(- 1j * X / Xi * y)
    W = np.exp(- 1j * X / Xi * y - Lambda_i * y - (1 / (2 * Xi**2)) *
               (np.sin(theta)**2 * (1 - np.cos(y)) + 1 / 2 * np.cos(theta)**2 * y**2))
    return acf  # , acf, imag_part
