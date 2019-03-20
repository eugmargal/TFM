import numpy as np
from scipy.stats import norm   #import gaussian cdf and pdf
#from scipy.optimize import brentq    # brentq method
from scipy.optimize import newton

def SwaptionNormalvol(FS,K,Annuity,Expiry,Premium,sign):
    """ Returns Bachelier (Normal) volatility from a swaption
        
            FS = Forward Swap Rate (%)
            K  = Strike Price (%)
       Annuity = in years
        Expiry = Swap's starting date
       Premium = Swap's premium (in %)
          sign = 1 if obtained from payer swaptions (calls)
                 -1 if obtaiend from receiver swaptions (puts)    
    """
    
    def d(sigma):
        return (FS-K)/(sigma*np.sqrt(Expiry))
    
    def Bnormal(sigma):
        return Premium/Annuity - (sign*(FS - K)*norm.cdf(sign*d(sigma)) + sigma*np.sqrt(Expiry)*norm.pdf(d(sigma)))
    
#    return brentq(Bnormal,0,10,xtol=2e-12)
    return newton(Bnormal,0.1)



def SwaptionLognvol(FS,K,Annuity,Expiry,Premium,s,sign):
    """ Returns Black (Lognormal) volatility from a swaption
        
            FS = Forward Swap Rate (%)
            K  = Strike Price (%)
       Annuity = in years
        Expiry = Swap's starting date
       Premium = Swap's premium (in %)
             s = shift (due to negative interest rates/strikes)
          sign = 1 if obtained from payer swaptions (calls)
                 -1 if obtained from receiver swaptions (puts)    
    """
    
    def d1(sigma):
        return (np.log((FS+s)/(K+s)) + 0.5*sigma**2*Expiry)/(sigma*np.sqrt(Expiry))
    
    def d2(sigma):    
        return (np.log((FS+s)/(K+s)) - 0.5*sigma**2*Expiry)/(sigma*np.sqrt(Expiry))
    
    def Blogn(sigma):
        if sign == 1:
            return Premium/Annuity - ((FS+s)*norm.cdf(d1(sigma)) - (K+s)*norm.cdf(d2(sigma)))
        elif sign == -1:
            return Premium/Annuity - ((K+s)*norm.cdf(-d2(sigma)) - (FS+s)*norm.cdf(-d1(sigma)))

#    return brentq(Blogn,0,1,xtol=2e-12)
    return newton(Blogn,0.4)