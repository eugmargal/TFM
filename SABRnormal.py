import numpy as np
from scipy.optimize import minimize


def normal_vol(k, f, t, Alpha, beta, Rho, Nu, shift):
    # shift both strike and forward ...
    """
        Given the parameters returns the parametrized volatilities
    """
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
    """
        Objective function to minimize for calibration while estimating
        the 3 parameters at the same time
    """
    sq_diff = []
    for i in range(len(k)):
        vol = normal_vol(k[i],f,t,param[0],beta,param[1],param[2],shift)
        sq_diff.append((vol-sigmaMKT[i])**2)

    return sum(sq_diff)

def calibrate(k,f,t,beta,shift,sigmaMKT,seed):
    """
        Calibration function for the estimation of the 3 parameters at once
    """
    bnd = ( (0, None), (-0.9999, 0.9999), (0, None)  )
    res = minimize(objfun, seed, args = (k,f,t,beta,shift,sigmaMKT), bounds = bnd, method = 'L-BFGS-B',options = {'ftol': 1e-16})
    return res.x

def fit_alpha_to_ATM(f, t, atm_vol, beta, rho, nu, shift):
    # shift  forward ...
    f += shift

    f_ = f ** (1 - beta)
    p = [
        - beta * (2 - beta) / (24 * f_ ** 2) * t * f ** beta,
        t * f ** beta * rho * beta * nu / (4 * f_),
        (1 + t * nu ** 2 * (2 - 3 * rho ** 2) / 24) * f ** beta,
        -atm_vol
    ]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    
    alpha_first_guess = atm_vol * f ** (-beta)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min]

def objfun2(param,k,f,t,beta,shift,sigmaMKT):
    """
        Objective function to minimize for calibration while estimating
        the 2 parameters at the same time and holding alpha as a function of them (ATM)
    """

    # CALIBRATE ALPHA TO ATM VOL ...        

    
    sq_diff = []; rho = param[0]; nu = param[1];
    atm_vol = sigmaMKT[np.nonzero(k == f)]
    alphaATM = fit_alpha_to_ATM(f, t, atm_vol, beta, rho, nu, shift)
    
    for i in range(len(k)):
        vol = normal_vol(k[i],f,t,alphaATM,beta,rho,nu,shift)
        sq_diff.append((vol-sigmaMKT[i])**2)
    return sum(sq_diff)

def calibrate2(k,f,t,beta,shift,sigmaMKT, seed):
    """
        Calibration function for the second method (estimating 2 parameters)
    """
    bnd = ( (-0.9999, 0.9999), (0.00001, None)  )
    res = minimize(objfun2, seed, args = (k,f,t,beta,shift,sigmaMKT), bounds = bnd, method = 'TNC',options = {'ftol': 1e-16})

    atm_vol = sigmaMKT[np.nonzero(k == f)]
 
    return [fit_alpha_to_ATM(f, t, atm_vol, beta, res.x[0], res.x[1], shift), res.x[0], res.x[1]]
    