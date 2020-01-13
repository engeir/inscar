# Generated with SMOP  0.41
from libsmop import *
# w_ion_gyro.m

    
@function
def w_ion_gyro(B=None,m_ion=None,*args,**kwargs):
    varargin = w_ion_gyro.varargin
    nargin = w_ion_gyro.nargin

    # Help text
#
    
    m_e=9.1e-31
# w_ion_gyro.m:5
    
    q_e=1.6e-19
# w_ion_gyro.m:6
    
    w_e=dot(q_e,B) / m_ion
# w_ion_gyro.m:8