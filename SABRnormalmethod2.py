import math
import numpy as np
from scipy.optimize import minimize

#### TRIAL ####
#K = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])/100;
#sigmaMKT = np.array([45.6, 41.6, 37.9, 36.6, 37.8, 39.2, 40.0])/100;
#FS = K[3];
#Expiry = 3;

def SABRNormalplot(param,FS,K,Expiry,beta,s):
    """ Returns SABR volatilies for each strike desired 
    
         INPUT:
         param = [alpha, rho, nu]
            FS = Forward Swap Rate (%)
            K  = Strike Price (%)
        Expiry = Swap's starting date
          beta = beta parameter
             s = SABR model shift
             
        OUTPUT:
  estimatedvol = list with the estimated vols for every K in the introduced array
    """
    FS = FS + s; K = K + s;
    logFK = np.log(FS/K); KFS = np.power(FS*K,(1-beta)/2);
    estimatedvol = []
    for j in range(len(K)):
        if  K[j] == FS:
            vol1 = (param[0]/FS**(1-beta)) * (1 + ((param[0]**2*(1-beta)**2)/(24*FS**(2-2*beta)) +
                   (param[0]*beta*param[1]*param[2])/(4*FS**(1-beta)) + (2-3*param[1]**2)*param[2]**2/24)*Expiry)
        
            estimatedvol.append(vol1)
        elif K[j] != FS:
            Aprimanum = 1 + (logFK[j])**2/24 + (logFK[j])**4/1920
            Aprimaden = 1 + ((1-beta)*logFK[j])**2/24 + ((1-beta)*logFK[j])**4/1920
            Bprima = (-param[0]**2*beta*(2-beta)**2)/(24*KFS[j]**2) + 0.25*param[0]*param[1]*param[2]*beta/KFS[j] + (2-3*param[1]**2)*param[2]**2/24
            c = param[2]*KFS[j]*logFK[j]/param[0]
            gprima = (math.sqrt(c**2 - 2*param[1]*c + 1) + c - param[1])/(1-param[1])
    
            vol2 = param[0]*(K[j]*FS)**(beta/2)*Aprimanum*(1+Bprima*Expiry)*c/(Aprimaden*math.log(gprima))

            estimatedvol.append(vol2)

    return estimatedvol

    
def SABRNormalfuncobj(param,FS,K,sigmaMKT,Expiry,beta,s):
    """ Objective function to minimize to calibrate the model
    
         INPUT:
         param = [rho, nu]
            FS = Forward Swap Rate (%)
            K  = Strike Price (%)
      sigmaMKT = vols observed in the market
        Expiry = Swap's starting date
          beta = beta parameter
             s = SABR model shift
    """
    FS = FS + s; K = K + s;
    logFK = np.log(FS/K); KFS = np.power(FS*K,(1-beta)/2); #print('logFK',logFK); print('KFS',KFS)
    sq_diff = []
    
    def alpha(rho,nu):
        raices = np.roots([(1-beta)**2*Expiry/(24*FS**(2-2*beta)),beta*rho*nu*Expiry/(4*FS**(1-beta)),
                  1 + (2-3*rho**2)*nu**2*Expiry/24,-sigmaMKT[np.nonzero(K == FS)]*FS**(1-beta)])
        raices = raices[np.isreal(raices) == 1] #returns only real numbers
        raiz = np.amin(raices[raices>0])     #returns minimum positive value
        return raiz.real

    
    for j in range(len(K)):
        if  K[j] == FS:
            vol1 = (alpha(param[0],param[1])/FS**(1-beta)) * (1 + ((alpha(param[0],param[1])**2*(1-beta)**2)/(24*FS**(2-2*beta)) +
                   (alpha(param[0],param[1])*beta*param[0]*param[1])/(4*FS**(1-beta)) + (2-3*param[0]**2)*param[1]**2/24)*Expiry)
        
            sq_diff.append((sigmaMKT[j]-vol1)**2)
            
        elif K[j] != FS:
            Aprimanum = 1 + (logFK[j])**2/24 + (logFK[j])**4/1920
            Aprimaden = 1 + ((1-beta)*logFK[j])**2/24 + ((1-beta)*logFK[j])**4/1920
            Bprima = (-alpha(param[0],param[1])**2*beta*(2-beta)**2)/(24*KFS[j]**2) + 0.25*alpha(param[0],param[1])*param[0]*param[1]*beta/KFS[j] + (2-3*param[0]**2)*param[1]**2/24
            c = param[1]*KFS[j]*logFK[j]/alpha(param[0],param[1])
            gprima = (math.sqrt(c**2 - 2*param[0]*c + 1) + c - param[0])/(1-param[0])
    
            vol2 = alpha(param[0],param[1])*(K[j]*FS)**(beta/2)*Aprimanum*(1+Bprima*Expiry)*c/(Aprimaden*math.log(gprima))

            sq_diff.append((sigmaMKT[j]-vol2)**2)
#    print(param,'\t',sum(sq_diff))
    return sum(sq_diff)
    
def SABRNormalcalb(FS,K,sigmaMKT,Expiry,beta,s):
    """ Returns SABR volatilies for each strike desired 
    
         INPUT:
            FS = Forward Swap Rate (%)
            K  = Strike Price (%)
      sigmaMKT = vols observed in the market
        Expiry = Swap's starting date
          beta = beta parameter
             s = SABR model shift
             
        OUTPUT:
           res = parameters obtained after optimizing SABRBlfuncobj
    """
    x0 = np.array([0.001,0.001]); bnd = ( (-0.9999, 0.9999), (0.0001, None)  )
    res = minimize(SABRNormalfuncobj,x0, args = (FS,K,sigmaMKT,Expiry,beta,s), bounds = bnd, method = 'L-BFGS-B',options = {'ftol': 1e-14})
    return res.x
