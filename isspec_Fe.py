# Generated with SMOP  0.41
from libsmop import *
# isspec_Fe.m


@function
def isspec_Fe(w=None,k=None,w_c=None,ny_e=None,Te=None,theta=None,*args,**kwargs):
    varargin = isspec_Fe.varargin
    nargin = isspec_Fe.nargin

    # ISSPEC_FE -

    kB=1.380662e-23
# isspec_Fe.m:5

    m_e=9.10938291e-31
# isspec_Fe.m:6

    m_p=1.672621778e-27
# isspec_Fe.m:7

    m_n=1.674927352e-27
# isspec_Fe.m:8

    # my_0    = 4*pi*1e-7;                # Permeability [Vs/Am]
# Ep_0    = 1/(my_0*c0^2);            # Permittivity [As/Vm]
# q_e     = 1.602176565e-19;        # elementary charge [C]

    X=sqrt(dot(m_e,w ** 2) / (dot(dot(dot(2,kB),Te),k ** 2)))
# isspec_Fe.m:15
    Xe=sqrt(dot(m_e,w_c ** 2) / (dot(dot(dot(2,kB),Te),k ** 2)))
# isspec_Fe.m:16
    Lambda_e=ny_e / w_c
# isspec_Fe.m:17
    if theta != 0:
        Fe_integrand=lambda y=None: exp(dot(dot(- 1j,(X / Xe)),y) - dot(Lambda_e,y) - dot((1 / (dot(2,Xe ** 2))),(dot(sin(theta) ** 2,(1 - cos(y))) + dot(dot(1 / 2,cos(theta) ** 2),y ** 2))))
# isspec_Fe.m:21
        Fe=quadgk(Fe_integrand,0,inf,'AbsTol',1e-16)
# isspec_Fe.m:25
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
  # b = ny_e/w_c (Really? looks odd to me??? BG 20161229)
  # c = kB*T*k^2/(m*w_c)
        a=w / w_c
# isspec_Fe.m:43
        b=ny_e / w_c
# isspec_Fe.m:44
        c=dot(dot(kB,Te),k ** 2) / (dot(m_e,w_c))
# isspec_Fe.m:45
        Fe=multiply(dot(sqrt(dot(2,pi)),exp(- a ** 2 / (dot(2,c)) + dot(dot(1j,a),b) / c + b ** 2 / (dot(2,c)))),(dot(Faddeeva_erf((- a + dot(1j,b)) / (dot(2,(- c / 2) ** (1 / 2)))),1j) + 1j)) / (dot(2,sqrt(- c / 2)))
# isspec_Fe.m:46

    Fe=1 - multiply((dot(1j,X) / Xe + Lambda_e),Fe)
# isspec_Fe.m:52
    # $$$ y = (1:1000)/10;
# $$$ subplot(2,1,1)
# $$$ plot(y,[real(Fe_integrand(y))])
# $$$ title(w/2/pi)
# $$$ subplot(2,1,2)
# $$$ plot(y,[imag(Fe_integrand(y))])
# $$$ drawnow