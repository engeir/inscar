# Generated with SMOP  0.41
from libsmop import *
# L_Debye.m

    
@function
def L_Debye(n_e=None,T_e=None,T_i=None,*args,**kwargs):
    varargin = L_Debye.varargin
    nargin = L_Debye.nargin

    # Help text
#
    
    Ep0=1e-09 / 36 / pi
# L_Debye.m:5
    q_e=1.6e-19
# L_Debye.m:6
    
    kB=1.38e-23
# L_Debye.m:7
    if nargin < 3:
        LD=sqrt(dot(dot(Ep0,kB),T_e) / (dot(max(0,n_e),q_e ** 2)))
# L_Debye.m:9
    else:
        LD=sqrt(dot(Ep0,kB) / ((max(0,n_e) / T_e + max(0,n_e) / T_i) / q_e ** 2))
# L_Debye.m:11
    