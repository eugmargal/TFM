
import numpy as np

def create_data(sample):
    shift = 0; beta = 0;
    
    #alpha, rho, nu
    params = np.random.random_sample((sample,3))*[0.01499, 1.96, 0.69999] + [0.00001, -0.98, 0.00001]
    
    T_index = [1/12, 1/4, 0.5] + [i for i in range(1,31)]
    T_random = np.random.randint(0,len(T_index),sample); T = []
    for i in range(sample):
        T.append(T_index[T_random[i]])
    
    
    F0 = np.random.random_sample(sample) * (0.035 + 0.005) - 0.005
    Kpb = np.linspace(-200,200,11); K = []
    for i in range(len(F0)):
        K.append(F0[i] + Kpb/10000)
    
    ### GENERATE THE SABR VOLS ###
        
    import SABRnormal
    vol_input = []
    for i in range(sample):
        vol_input_ind = []
        for j in range(len(Kpb)):
            aux = SABRnormal.normal_vol(K[i][j],F0[i],T[i],params[i][0],beta,params[i][1],params[i][2],shift)
            vol_input_ind.append(aux)
        vol_input.append(vol_input_ind)
    
    ### Create the matrix of inputs ###
    K = np.matrix(K); vol_input = np.matrix(vol_input)
    # The order is: [K (vector of strikes), sigma (vector of vols), T]
    X = np.hstack((K,vol_input,np.matrix(T).T))
    y = params
    np.save('X_sample.npy',X); np.save('y_sample.npy',y)
    return X, y

#X,y = create_data(500000);
X = np.load('X_sample.npy'); y = np.load('y_sample.npy')

# Create data input and output for neural network
# The output to predict will be the parameters alpha, rho, nu
#X_mean = X.mean(axis = 0); X_std = X.std(axis = 0)
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
y[:,0] = np.ravel(scaler.fit_transform(y[:,0].reshape(-1,1)))

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
#X_train = preprocessing.scale(X_train)
#X_test = preprocessing.scale(X_test)



### NEURAL NETWORK ###
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

def build_keras( hidden_layers = 3, size_layers = 48):
    model = Sequential()
    model.add(Dense(size_layers, activation='elu', input_dim = 23, kernel_initializer='he_uniform'))
    for i in range(hidden_layers):
        model.add(Dense(size_layers, activation = 'elu', kernel_initializer= 'he_uniform'))
    model.add(Dense(3))
    
    sgd = keras.optimizers.Adam(lr=0.001)
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=sgd,
              metrics=['accuracy'])  
    return model


earlystop = EarlyStopping(monitor = 'val_loss', patience=40, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=15, cooldown = 1, min_lr= 0.000009, verbose = 1)

# Grid search of best hyperparameters (nº hidden layers, nº units)
def searchgrid(X_train, y_train, X_test, y_test):

    model_keras = KerasRegressor(build_fn = build_keras, verbose  = 1)

    keras_fit_params = {
            'callbacks': [reduce_lr],
            'epochs': 1,   # number epochs the set (from cv) is trained
            'batch_size': 128,
            'validation_data': (X_test, y_test),
            'verbose': 1}
    
    # define grid search parameters
    hidden_layers_grid = [4 ,5 , 6, 7, 8]
    layer_size_grid = [32, 64, 128, 256]
    param_grid = dict(hidden_layers = hidden_layers_grid,
                      size_layers = layer_size_grid)
    
#    hidden_layers_grid = [4 ,5 , 6, 7, 8]
#    layer1size = [256, 128, 64, 32]
#    for i in 
    
    
    from sklearn.model_selection import GridSearchCV   
    gs_keras = GridSearchCV( 
        estimator = model_keras, 
        param_grid = param_grid,
        fit_params = keras_fit_params,
        n_jobs = 1,
        verbose = 1)

    gs_keras.fit(X_train, y_train)

    print("Best: %f using %s" % (gs_keras.best_score_, gs_keras.best_params_))
#    means = gs_keras.cv_results_['mean_test_score']
#    stds = gs_keras.cv_results_['std_test_score']
#    params = gs_keras.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))
    best_result = []
    for param, value in gs_keras.best_params_.items():
        best_result.append([param, value])
    
    return best_result #gs_keras

#config = searchgrid(X_train, y_train, X_test, y_test)
#np.save('result.npy',config); 


#model = build_keras(config[0][1],config[1][1])
model = build_keras(6,64)

history = model.fit(X_train, y_train,
          batch_size = 128,
          epochs = 200,
          verbose = 1,
          validation_data= (X_test, y_test),
          callbacks = [reduce_lr])


#model.save_weights('volstoparams_weights_normal.h5')
model.save('volstoparams_scale.h5')
#NNParameters=[]
#for i in range(1,len(model.layers)):
#    NNParameters.append(model.layers[i].get_weights())
    
#y_test[:,0] = np.ravel(scaler.inverse_transform(y_test[:,0].reshape(-1,1)))
# Comprobar con modelo aleatorio #
import SABRnormal
import matplotlib.pyplot as plt
testcase = -2000; Xtestcase = np.ravel(X_test[testcase,:]);
testparams = np.ravel(model.predict(X_test[testcase,:].reshape(1,-1)))
testparams[0] = scaler.inverse_transform(testparams[0])
vol_test = []; 
for j in range(11):
    aux = SABRnormal.normal_vol(Xtestcase[j],Xtestcase[5],Xtestcase[-1],testparams[0],0,testparams[1],testparams[2],0)
    vol_test.append(aux)

plt.figure()
plt.subplot(121)
plt.plot(Xtestcase[:11], Xtestcase[11:-1], color = 'blue', marker = 'x', linewidth = 1.0);
plt.plot(Xtestcase[:11],np.array(vol_test), color = 'green', marker = 'o', linewidth = 1.0);
plt.grid(True); plt.legend(['Approx','ANN']); plt.title('volatility smile')

plt.subplot(122)
plt.plot(Xtestcase[:11], (Xtestcase[11:-1] - np.squeeze(vol_test))/Xtestcase[11:-1], linewidth = 1.0);
plt.grid(True); plt.legend(['error']); plt.title('approximation error')
plt.gca().set_yticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_yticks()]) #vol as %
plt.tight_layout()


"""
F0 = 0.03; Ktest = F0 + Kpb/10000; T = 20; shift = 0;  beta = 0.3;

rho = 0.54; nu = 0.155; alphaaux = np.linspace(0.001,0.015, 200); param0 = []
for i in range(len(alphaaux)):
    vol_input_ind = []
    for j in range(len(Kpb)):
        aux = SABRnormal.normal_vol(Ktest[j],F0,T,alphaaux[i],beta,rho,nu,shift)
        vol_input_ind.append(aux)
    Xaux = np.hstack((Ktest,F0,vol_input_ind,T))
    Xaux = Xaux.reshape(1,-1)
    Xaux = (Xaux - X_mean)/X_std
    paramspred = model.predict(Xaux)
    param0.append(paramspred[0][0])

alpha = 0.01; nu = 0.3; rhoaux = np.linspace(-0.98,0.98, 200); param1 = []
for i in range(len(rhoaux)):
    vol_input_ind = []
    for j in range(len(Kpb)):
        aux = SABRnormal.normal_vol(Ktest[j],F0,T,alpha,beta,rhoaux[i],nu,shift)
        vol_input_ind.append(aux)
    Xaux = np.hstack((Ktest,F0,vol_input_ind,T))
    Xaux = Xaux.reshape(1,-1)
    Xaux = (Xaux - X_mean)/X_std
    paramspred = model.predict(Xaux)
    param1.append(paramspred[0][1])

alpha = 0.01; rho = 0.44; nuaux = np.linspace(0.01,0.7, 200); param2 = []
for i in range(len(nuaux)):
    vol_input_ind = []
    for j in range(len(Kpb)):
        aux = SABRnormal.normal_vol(Ktest[j],F0,T,alpha,beta,rho,nuaux[i],shift)
        vol_input_ind.append(aux)
    Xaux = np.hstack((Ktest,F0,vol_input_ind,T))
    Xaux = Xaux.reshape(1,-1)
    Xaux = (Xaux - X_mean)/X_std
    paramspred = model.predict(Xaux)
    param2.append(paramspred[0][2])


plt.subplot(311)
plt.plot(alphaaux,np.abs((param0-alphaaux)/param0),'o-'); plt.title('Relative error in alpha');
plt.yscale('log')

plt.subplot(312)
plt.plot(rhoaux,np.abs((param1-rhoaux)/param1),'o-'); plt.title('Relative error in rho');
plt.yscale('log')

plt.subplot(313)
plt.plot(nuaux,np.abs((param2-nuaux)/param2),'o-'); plt.title('Relative error in nu');
plt.yscale('log')
plt.tight_layout()
plt.show()
"""
