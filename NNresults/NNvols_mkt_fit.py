import numpy as np
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
    Xmarket[i,:] = np.hstack((strike, vol, T[11*i]))

from keras import backend as K
K.set_floatx('float64')     # Set double precision so everithing works fine in calibration ...
from keras.models import load_model


X5_ = Xmarket[ Xmarket[:,-1] >= 5 ]   # Select only market values with maturity >= 5Y
#X_5 = Xmarket[ Xmarket[:,-1] < 5 ]    # Select only market values with maturity < 5Y
#X_1 = Xmarket[ Xmarket[:,-1] < 1]     # Select only market values with maturity < 1Y

#NN = load_model('paramstovols_fl64.h5')
NN5_30 = load_model('paramstovols_fl64_5_30.h5')
#NN_5 = load_model('paramstovols_fl64_5.h5')
#X = np.load('X_sample_vols.npy'); vols = np.load('y_sample_vols.npy');

def analisis(X, network, seed, plot_swap_option, plot_option, test_Hagan, test_Hagan_ATM):
    
    def objfun(param,k,f,t,beta,shift,sigmaMKT):
        """
            Objective function to minimize for calibration while estimating
            the 3 parameters at the same time
        """
        vbles = np.hstack((k,param,t)).reshape(1,-1)
        vols = np.ravel(network.predict(vbles))
        MSE = np.sum((vols - sigmaMKT)**2)     
        return MSE

    import SABRnormal
    from scipy.optimize import minimize

    number_swaptions = X.shape[0]
    error_test = np.zeros((number_swaptions,11))
    
    # calibration to NN module
    for i in range(number_swaptions):
        args = ( X[i,:11],X[i,5],X[i,-1],0,0,X[i,11:-1] )
        res = minimize(objfun, seed, args,
                           method = 'BFGS')   #,options = {'disp': True})           
        vol_test = []; 
        for j in range(11):
            aux = SABRnormal.normal_vol(X[i,j],X[i,5],X[i,-1],res.x[0],0,res.x[1],res.x[2],0)
            vol_test.append(aux)
        error_test[i] = (X[i,11:-1] - vol_test)
    RMSE = np.sqrt(np.mean(np.square(error_test), axis = 0))*10000

    title = ["ATM - 5", "ATM - 4", "ATM - 3", "ATM - 2", "ATM - 1", "ATM", "ATM + 1", "ATM + 2", "ATM + 3 ", "ATM + 4", "ATM + 5"];
    data = pd.DataFrame(RMSE,columns = ['NN'], index = title)
    
    import matplotlib.pyplot as plt

    ### 3D plot of RMSE vs maturity of swaptions and options ###
    if plot_swap_option == 1:
        x_array = np.unique(X[:,-1])
        result = np.sqrt(np.mean(np.square(error_test), axis = 1))       
        z = np.hsplit(result,14)
        y_array = np.hstack((np.linspace(1,10,10), 15,20,25,30))
        x_array, y_array = np.meshgrid(x_array, y_array)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_array, y_array, np.array(z),
                        cmap='viridis', edgecolor='none')
        ax.set_title('Volatilility error mean accross strikes');
        ax.set_xlabel('Option maturity'); ax.set_ylabel('Swap maturity')
        ax.set_zlabel('RMSE mean')
        plt.show()
        
    ### Plot of RMSE vs swaption maturity ###
    if plot_option == 1:
        x_array = np.unique(X[:,-1])
        result = np.sqrt(np.mean(np.square(error_test), axis = 1))       
        z = np.hsplit(result,14)
        plt.figure()
        plt.plot(x_array,np.mean(np.matrix(z),axis=0).reshape(-1,1))
        plt.title('RMSE accross maturities'); plt.xlabel('Swaption maturities'); plt.ylabel('RMSE')
    
    if test_Hagan == 1:
        error_Hagan = np.zeros((number_swaptions,11))
        for i in range(len(X)):
            res = SABRnormal.calibrate(X[i,:11],X[i,5],X[i,-1],0,0,X[i,11:-1],seed)
            vol_test = []; 
            for j in range(11):
                aux = SABRnormal.normal_vol(X[i,j],X[i,5],X[i,-1],res[0],0,res[1],res[2],0)
                vol_test.append(aux)
            error_Hagan[i] = (X[i,11:-1] - vol_test)
        
        RMSE_Hagan = np.sqrt(np.mean(np.square(error_Hagan), axis = 0))*10000
        result_Hagan = pd.DataFrame(RMSE_Hagan, columns = ['Hagan'], index = title)
        data = data.add(result_Hagan, fill_value = 0)
        
    if test_Hagan_ATM == 1:
        error_Hagan_ATM = np.zeros((number_swaptions,11))
        seed = np.delete(seed,0)     # now alpha is implicit, no need to fit it
        for i in range(len(X)):
            res = SABRnormal.calibrate2(X[i,:11],X[i,5],X[i,-1],0,0,X[i,11:-1],seed)
            vol_test = []; 
            for j in range(11):
                aux = SABRnormal.normal_vol(X[i,j],X[i,5],X[i,-1],res[0],0,res[1],res[2],0)
                vol_test.append(aux)
            error_Hagan_ATM[i] = (X[i,11:-1] - vol_test)       

        RMSE_Hagan_ATM = np.sqrt(np.mean(np.square(error_Hagan_ATM), axis = 0))*10000
        result_Hagan_ATM = pd.DataFrame(RMSE_Hagan_ATM, columns = ['Hagan ATM'], index = title)
        data = data.add(result_Hagan_ATM, fill_value = 0)
                
    return data[['NN','Hagan','Hagan ATM']]

errores = analisis(X5_, NN5_30, [0.001, 0.01, 0.4] , 0,0,1,1)

pd.options.display.float_format = '{:.4f}'.format
print(errores[['NN','Hagan','Hagan ATM']])

#pd.reset_option('display.float_format')

 
        
        
        
        
        
        