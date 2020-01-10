# Generated with SMOP  0.41
from libsmop import *
# y_integrand.m

    
@function
def y_integrand(y=None,w=None,k=None,w_g=None,ny_coll=None,kBT=None,m=None,theta=None,*args,**kwargs):
    varargin = y_integrand.varargin
    nargin = y_integrand.nargin

    # Y_INTEGRAND - 
#
    
    f=exp(dot(dot(- 1j,w) / w_g,y) - dot(ny_coll / w_g,y) - dot(dot(kBT,k) / (dot(m,w_g ** 2)),(dot(sin(theta) ** 2,(1 - cos(y))) + dot(y ** 2 / 2,cos(theta)))))
# y_integrand.m:5