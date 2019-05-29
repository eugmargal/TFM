import numpy as np
from keras.models import load_model
model = load_model('paramstovols.h5')
X = np.load('X_sample_vols.npy'); vols = np.load('y_sample_vols.npy');
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objfun(param,k,f,t,beta,shift,sigmaMKT):
    """
        Objective function to minimize for calibration while estimating
        the 3 parameters at the same time
    """
    alpha,rho,nu = param
    inputs = np.hstack((k, alpha, rho, nu,t)).reshape(1,-1)
    vols = np.ravel(model.predict(inputs))
    MSE = np.sum((vols - sigmaMKT)**2)

    return MSE

def calibrate(k,f,t,beta,shift,sigmaMKT,seed):
    """
        Calibration function for the estimation of the 3 parameters at once
    """
    bnd = ( (0, None), (-0.9999, 0.9999), (0, None)  )
    res = minimize(objfun, seed, args = (k,f,t,beta,shift,sigmaMKT), bounds = bnd,
                   method = 'L-BFGS-B', options={'disp': True})
    return res.x

seed = [0.004,0.5,0.4]
#param1 = calibrate(X[0,:11],X[0,5],X[0,-1],0,0,vols[0,:],seed)

bnd = ( (0, None), (-0.9999, 0.9999), (0, None)  )
args = ( X[0,:11],X[0,5],X[0,-1],0,0,vols[0,:] )


res = minimize(objfun, seed, args, bounds = bnd,
               method = 'L-BFGS-B',options = {'ftol': 1e-16, 'disp': True})


import SABRnormal
vol_test = []; 
for j in range(11):
    aux = SABRnormal.normal_vol(X[0,j],X[0,5],X[0,-1],res.x[0],0,res.x[1],res.x[2],0)
    vol_test.append(aux)

plt.figure()
plt.subplot(121)
plt.plot(X[0,:11], vols[0,:], color = 'blue', marker = 'x', linewidth = 1.0);
plt.plot(X[0,:11],vol_test, color = 'green', marker = 'o', linewidth = 1.0);
plt.grid(True); plt.legend(['Approx','ANN']); plt.title('volatility smile')

plt.subplot(122)
plt.plot(X[0,:11], (vols[0,:] - vol_test), linewidth = 1.0);
plt.grid(True); plt.legend(['error']); plt.title('approximation error')
#plt.gca().set_yticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_yticks()]) #vol as %
plt.tight_layout()