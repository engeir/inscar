# Generated with SMOP  0.41
from libsmop import *
# w_plasma.m

    
@function
def w_plasma(n_e=None,*args,**kwargs):
    varargin = w_plasma.varargin
    nargin = w_plasma.nargin

    # Help text
#
    
    Ep0=1e-09 / 36 / pi
# w_plasma.m:5
    m_e=9.1e-31
# w_plasma.m:6
    
    q_e=1.6e-19
# w_plasma.m:7
    
    w_e=sqrt(dot(max(0,n_e),q_e ** 2) / m_e)
# w_plasma.m:9