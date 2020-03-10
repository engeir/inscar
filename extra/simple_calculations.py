import ast

from PyPDF2 import PdfFileReader
import scipy.constants as const
import numpy as np

def find(path):
    with open(path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()

    keywords = ast.literal_eval(information['/Keywords'])
    return keywords

def ions(path):
    keys = find(path)
    k0 = 2 * keys['F0'] * 2 * np.pi / const.c  # Radar wavenumber
    v_th = abs(const.k * keys['T_E'] / (keys['MI'] * const.m_p))**.5
    v_th_e = np.sqrt(const.k * keys['T_E'] / const.m_e)
    # l_D = np.sqrt(const.epsilon_0 * const.k / ((const.m_e / keys['T_E'] + const.m_e / keys['T_I']) / const.e**2))
    # l_D = np.sqrt(const.epsilon_0 * const.k * keys['T_E'] / (keys['NE'] * const.elementary_charge**2))
    l_D = np.sqrt(const.epsilon_0 * const.k * keys['T_I'] / (keys['NE'] * const.elementary_charge**2))
    # a = 1 / l_De**2 + 1 / l_Di**2
    # l_D = np.sqrt(1 / a)
    w = k0 * v_th / (1 + k0**2 * l_D**2)  # / (2 * np.pi)
    print(f'ion thermal velocity = {v_th:1.3e} = {v_th/v_th_e:1.3e}v_th_e\nfrequency = {w:1.3e}')

def electrons(path):
    keys = find(path)
    k0 = 2 * keys['F0'] * 2 * np.pi / const.c  # Radar wavenumber
    v_th = np.sqrt(const.k * keys['T_E'] / const.m_e)
    # l_D = np.sqrt(const.epsilon_0 * const.k * keys['T_E'] / (keys['NE'] * const.elementary_charge**2))
    w_pe = np.sqrt(keys['NE'] * const.elementary_charge **
                   2 / (const.m_e * const.epsilon_0))
    w = np.sqrt(w_pe**2 + 3 * k0**2 * v_th**2) / (2 * np.pi)
    print(f'electron thermal velocity = {v_th:1.3e}\nfrequency = {w:1.3e}')
