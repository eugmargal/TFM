import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def normal_vol(k, f, t, Alpha, beta, Rho, Nu, shift):
    # shift both strike and forward ...
    f = f + shift
    k = k + shift
    
    tol = 0.0000000001

    f_av = 0.5 * (f + k)
    gamma_1 = beta / (f_av )
    gamma_2 = -1.0 * beta * (1 - beta) / ((f_av ) * (f_av )) 
    zeta = (Nu / Alpha) * (f - k) / (f_av**beta)
    int_denom = (1 / (1 - beta)) * (f ** (1 - beta)  - k **(1 - beta) )

    x_cir = np.log((np.sqrt(1 - 2 * Rho * zeta + zeta *zeta) + zeta - Rho) / (1 - Rho))
    
    SABRTerms = 1.0 + ((2.0 * gamma_2 - gamma_1 *gamma_1) / 24.0 * Alpha * Alpha * (f_av ** (2.0 * beta ) ) + 
      (1.0 / 4.0) * Rho * Nu * Alpha * gamma_1 * (f_av ** beta) + 
      (2.0 - 3.0 * Rho * Rho) * Nu * Nu / 24.0) * t
    
    if np.abs( zeta ) <= tol:
      Vol = Alpha * (f ** beta) * SABRTerms
    else :
      Vol = Alpha * ((f - k) / int_denom) * (zeta / x_cir) * SABRTerms
    
    return Vol

def objfun(param,k,f,t,beta,shift,sigmaMKT):
    sq_diff = []
    for i in range(len(k)):
        vol = normal_vol(k[i],f,t,param[0],beta,param[1],param[2],shift)
        sq_diff.append((vol-sigmaMKT[i])**2)
    return sum(sq_diff)

def calibrate(k,f,t,beta,shift,sigmaMKT):
    x0 = np.array([sigmaMKT[np.nonzero(k == f)],0,0.1]); bnd = ( (0.00001, None), (-0.9999, 0.9999), (0.0001, None)  )
    res = minimize(objfun,x0, args = (k,f,t,beta,shift,sigmaMKT), bounds = bnd, method = 'L-BFGS-B',tol = 1e-10)#,options = {'ftol': 1e-16})
    return res.x


def objfun2(param,k,f,t,beta,shift,sigmaMKT):
    
    def alpha(rho,nu):
#        gamma_1 = beta / f
#        gamma_2 = -1.0 * beta * (1 - beta) / (f * f)         
#        coef1 = (2*gamma_2 - gamma_1 * gamma_1) * (f ** (2*beta)) * t / 24
#        coef2 = gamma_1 * rho * nu * (f ** beta) * t / 4
#        coef3 = 1 + (2 - 3 * rho ** 2) * (nu ** 2) * t / 24
#        coef4 = - sigmaMKT[np.nonzero(k == f)] * f ** (-beta)
        coef1 = beta*(2-beta)*t/(24*f**(2-2*beta))
        coef2 = rho*beta*nu*t/(4*f**(1-beta))
        coef3 = 1 + (2 - 3 * rho * rho) * nu * nu * t / 24
        coef4 = - sigmaMKT[np.nonzero(k == f)] * f ** (- beta)
        raices = np.roots([coef1,coef2,coef3,coef4])
        raices = raices[np.isreal(raices) == 1] #returns only real numbers
        raiz = np.amin(raices[raices>0])     #returns minimum positive value
        return raiz.real
    
    sq_diff = []
    
    for i in range(len(k)):
        vol = normal_vol(k[i],f,t,alpha(param[0],param[1]),beta,param[0],param[1],shift)
        sq_diff.append((vol-sigmaMKT[i])**2)
    print(sum(sq_diff),param)
    return sum(sq_diff)

def calibrate2(k,f,t,beta,shift,sigmaMKT):
    x0 = np.array([0,0.1]); bnd = ( (-0.9999, 0.9999), (0.00001, None)  )
    res = minimize(objfun2,x0, args = (k,f,t,beta,shift,sigmaMKT), bounds = bnd, method = 'L-BFGS-B', tol = 1e-8)
    return res.x
    