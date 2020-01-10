# Generated with SMOP  0.41
from libsmop import *
# isspec_ro.m

    
@function
def isspec_ro(f=None,f0=None,Ne=None,Te=None,Nu_e=None,mi=None,Ti=None,Nu_i=None,B=None,theta=None,*args,**kwargs):
    varargin = isspec_ro.varargin
    nargin = isspec_ro.nargin

    # ISSPEC_RO - 
#
    
    c0=299792458
# isspec_ro.m:5
    
    m_e=9.10938291e-31
# isspec_ro.m:6
    
    m_p=1.672621778e-27
# isspec_ro.m:7
    
    m_n=1.674927352e-27
# isspec_ro.m:8
    
    q_e=1.602176565e-19
# isspec_ro.m:9
    
    kB=1.380662e-23
# isspec_ro.m:11
    
    my_0=dot(dot(4,pi),1e-07)
# isspec_ro.m:12
    
    Ep_0=1 / (dot(my_0,c0 ** 2))
# isspec_ro.m:13
    
    w=dot(dot(f,2),pi)
# isspec_ro.m:18
    w0=dot(dot(f0,2),pi)
# isspec_ro.m:19
    M_i=dot(mi,(m_p + m_n)) / 2
# isspec_ro.m:21
    
    # there is an equal number oh
                      # protons and neutrons, sue me...
    
    k0=w0 / c0
# isspec_ro.m:25
    w_c=w_e_gyro(norm(B))
# isspec_ro.m:26
    W_c=w_ion_gyro(norm(B),dot(mi,m_p))
# isspec_ro.m:27
    w_p=w_plasma(Ne)
# isspec_ro.m:28
    l_D=L_Debye(Ne,Te)
# isspec_ro.m:29
    Xp=(dot(m_e,w_p ** 2) / (dot(dot(dot(2,kB),Te),k0 ** 2)))
# isspec_ro.m:30
    Xp=sqrt(1 / (dot(dot(2,l_D ** 2),k0 ** 2)))
# isspec_ro.m:31
    for i_w in arange(1,length(w)).reshape(-1):
        Fe=isspec_Fe(w(i_w),k0,w_c,Nu_e,Te,theta)
# isspec_ro.m:33
        Fi=isspec_Fi(w(i_w),k0,W_c,Nu_i,Ti,theta,mi)
# isspec_ro.m:34
        Is[i_w]=dot(dot(q_e ** 2,Ne) / pi / w(i_w),(imag(- Fe) + imag(- Fi))) / abs(1 + dot(dot(2,Xp ** 2),(Fe + Fi))) ** 2
# isspec_ro.m:36
    