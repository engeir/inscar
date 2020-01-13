# Generated with SMOP  0.41
from libsmop import *
# w_e_gyro.m

    
@function
def w_e_gyro(B=None,*args,**kwargs):
    varargin = w_e_gyro.varargin
    nargin = w_e_gyro.nargin

    # Help text
#
    
    m_e=9.1e-31
# w_e_gyro.m:5
    
    q_e=1.6e-19
# w_e_gyro.m:6
    
    w_e=dot(q_e,B) / m_e
# w_e_gyro.m:8