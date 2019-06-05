import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Smile_30_4_2019.csv', sep = ',', header = 0, index_col = False, engine = 'python')
data[['Currency','Maturity','IRS_duration']] = data.Reference.str.split("_", expand = True)
matur = data.Maturity.str.extract('(\d+)([MY])', expand = True)
T = []

for i in range(len(matur)):
    if matur.iloc[i,1] == 'M':
        aux = pd.to_numeric(matur.iloc[i,0]) / 12
        T.append(aux)
    else:
        T.append(matur.iloc[i,0])
        
def strikeABS(K_bps, FR):
    K = FR + K_bps/10000
    return K

length = int(len(data) / 11); Xmarket = np.zeros((length,23))
for i in range(length):
    strike = []; vol = []
    for j in range(11):
        Kaux = strikeABS(pd.to_numeric(data.StrikeATMdiff[j + 11*i]),pd.to_numeric(data.Forward_Rate[j + 11*i]))
        strike.append(Kaux)
        vol.append(pd.to_numeric(data.Volatility[j + 11*i]))
    Xmarket[i,:] = np.hstack((strike,vol, T[11*i]))

from keras import backend as K
# Set double precision so everithing works fine in calibration ...
K.set_floatx('float64')
from keras.models import load_model
NN = load_model('paramstovols_fl64.h5')
#X = np.load('X_sample_vols.npy'); vols = np.load('y_sample_vols.npy');
import SABRnormal

from scipy.optimize import minimize
def objfun(param,k,f,t,beta,shift,sigmaMKT):
    """
        Objective function to minimize for calibration while estimating
        the 3 parameters at the same time
    """
    vbles = np.hstack((k,param,t)).reshape(1,-1)
    vols = np.ravel(NN.predict(vbles))
    MSE = np.sum((vols - sigmaMKT)**2)

    return MSE


### Test and show plot of only one swaption ###
seed = [0.001,0,0.01]; testcase = 77
bnd = ( (0, None), (-0.9999, 0.9999), (0, None)  )
args = ( Xmarket[testcase,:11],Xmarket[testcase,5],Xmarket[testcase,-1],0,0,Xmarket[testcase,11:-1] )


res = minimize(objfun, seed, args, bounds = bnd,
               method = 'TNC',options = {'ftol': 1e-16, 'disp': True})

vol_test = []; 
for j in range(11):
    aux = SABRnormal.normal_vol(Xmarket[testcase,j],Xmarket[testcase,5],Xmarket[testcase,-1],res.x[0],0,res.x[1],res.x[2],0)
    vol_test.append(aux)
    
plt.figure()
plt.subplot(121)
plt.plot(Xmarket[testcase,:11],Xmarket[testcase,11:-1], color = 'blue', marker = 'x', linewidth = 1.0);
plt.plot(Xmarket[testcase,:11],vol_test, color = 'green', marker = 'o', linewidth = 1.0);
plt.grid(True); plt.legend(['Market','ANN']); plt.title('volatility smile')

plt.subplot(122)
plt.plot(Xmarket[testcase,:11], (Xmarket[testcase,11:-1] - vol_test), linewidth = 1.0);
plt.grid(True); plt.legend(['error']); plt.title('approximation error')
#plt.gca().set_yticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_yticks()]) #vol as %
plt.tight_layout()


### Test MSE error of all market swaptions ###
error_test = np.zeros((252,11))
for i in range(len(Xmarket)):
    args = ( Xmarket[i,:11],Xmarket[i,5],Xmarket[i,-1],0,0,Xmarket[i,11:-1] )
    res = minimize(objfun, seed, args, bounds = bnd,
               method = 'TNC',options = {'ftol': 1e-16, 'disp': True})
    vol_test = []; 
    for j in range(11):
        aux = SABRnormal.normal_vol(Xmarket[i,j],Xmarket[i,5],Xmarket[i,-1],res.x[0],0,res.x[1],res.x[2],0)
        vol_test.append(aux)
    error_test[i] = (Xmarket[i,11:-1] - vol_test)
RMSE = np.sqrt(np.sum(np.square(error_test), axis = 0)/len(Xmarket))
title = np.linspace(-5,5,11)
result = pd.DataFrame(RMSE.reshape(1,-1),index = None, columns = title)

### 3D plot of RMSE vs maturity of swaptions and options ###
plt.figure()
result2 = np.vstack((np.sqrt(np.mean(np.square(error_test), axis = 1)), Xmarket[:,-1]))  
X = result2[1,:18]
Y = np.hstack((np.linspace(1,10,10), 15,20,25,30))
X, Y = np.meshgrid(X, Y)
z = np.hsplit(result2[0,:],14)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y, np.array(z),
                cmap='viridis', edgecolor='none')
ax.set_title('Volatilility error mean accross strikes');
ax.set_xlabel('Option maturity'); ax.set_ylabel('Swap maturity')
ax.set_zlabel('RMSE mean')
plt.show()

### Plot of RMSE vs swaption maturity ###
plt.figure()
plt.plot(result2[1,:18],np.sum(np.matrix(z),axis=0).reshape(-1,1))
plt.title('RMSE accross maturities'); plt.xlabel('Swaption maturities'); plt.ylabel('RMSE')



