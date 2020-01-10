# Generated with SMOP  0.41
from libsmop import *
# isspec_Fi.m

    
@function
def isspec_Fi(w=None,k=None,w_c=None,ny_i=None,Ti=None,theta=None,Mi=None,*args,**kwargs):
    varargin = isspec_Fi.varargin
    nargin = isspec_Fi.nargin

    # ISSPEC_Fi -
    
    kB=1.380662e-23
# isspec_Fi.m:6
    
    # my_0    = 4*pi*1e-7;                # Permeability [Vs/Am]
# Ep_0    = 1/(my_0*c0^2);            # Permittivity [As/Vm]
# q_e     = 1.602176565e-19;        # elementary charge [C]
    m_e=9.10938291e-31
# isspec_Fi.m:10
    
    m_p=1.672621778e-27
# isspec_Fi.m:11
    
    m_n=1.674927352e-27
# isspec_Fi.m:12
    
    M_i=dot(Mi,(m_p + m_n)) / 2
# isspec_Fi.m:15
    
    # there is an equal number oh
                      # protons and neutrons, sue me...
    X=sqrt(dot(M_i,w ** 2) / (dot(dot(dot(2,kB),Ti),k ** 2)))
# isspec_Fi.m:18
    Xi=sqrt(dot(M_i,w_c ** 2) / (dot(dot(dot(2,kB),Ti),k ** 2)))
# isspec_Fi.m:19
    Lambda_i=ny_i / w_c
# isspec_Fi.m:21
    if theta != 0:
        Fi_integrand=lambda y=None: exp(dot(dot(- 1j,(X / Xi)),y) - dot(Lambda_i,y) - dot((1 / (dot(2,Xi ** 2))),(dot(sin(theta) ** 2,(1 - cos(y))) + dot(dot(1 / 2,cos(theta) ** 2),y ** 2))))
# isspec_Fi.m:25
        Fi=quadgk(Fi_integrand,0,inf,'AbsTol',1e-16)
# isspec_Fi.m:30
    else:
        # Analytical solution to the integral
  #             /     2             2 \
  #             |    a    a b i    b  | /    /    a - b i    \       \
  # sqrt(pi) exp| - --- + ----- + --- | | erf| ------------- | i + i |
  #             \   2 c     c     2 c / |    |       /   c \ |       |
  #                                     |    | 2 sqrt| - - | |       |
  #                                     \    \       \   2 / /       /
  # ------------------------------------------------------------------
  #                                  /   c \
  #                            2 sqrt| - - |
  #                                  \   2 /
        # Where
  # a = w/w_c
  # b = ny_i/w_c (Really? looks odd to me??? BG 20161229)
  # c = kB*T*k^2/(m*w_c)
        a=w / w_c
# isspec_Fi.m:48
        b=ny_i / w_c
# isspec_Fi.m:49
        c=dot(dot(kB,Ti),k ** 2) / (dot(M_i,w_c))
# isspec_Fi.m:50
        Fi=multiply(dot(sqrt(dot(2,pi)),exp(- a ** 2 / (dot(2,c)) + dot(dot(1j,a),b) / c + b ** 2 / (dot(2,c)))),(dot(Faddeeva_erf((- a + dot(1j,b)) / (dot(2,(- c / 2) ** (1 / 2)))),1j) + 1j)) / (dot(2,sqrt(- c / 2)))
# isspec_Fi.m:51
    
    Fi=1 - dot((dot(1j,X) / Xi + Lambda_i),Fi)
# isspec_Fi.m:58
    # $$$ if w/2/pi == -1e4
# $$$   keyboard
# $$$ end
# $$$ y = (1:1000)/10;
# $$$ subplot(2,1,1)
# $$$ plot(y,[real(Fi_integrand(y))])
# $$$ title(w/2/pi)
# $$$ subplot(2,1,2)
# $$$ plot(y,[imag(Fi_integrand(y))])
# $$$ drawnow