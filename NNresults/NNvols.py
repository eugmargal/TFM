
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
    X = np.hstack((K,params,np.matrix(T).T))
    y = vol_input
    np.save('X_sample_vols.npy',X); np.save('y_sample_vols.npy',y)
    return X, y

def build_keras( hidden_layers = 3, size_layers = 48):
    model = Sequential()
    model.add(Dense(size_layers, activation='elu', input_dim = 15, kernel_initializer='he_uniform'))
    for i in range(hidden_layers):
        model.add(Dense(size_layers, activation = 'elu', kernel_initializer= 'he_uniform'))
    model.add(Dense(11))
    
    sgd = keras.optimizers.Adam(lr=0.001)
    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=sgd,
              metrics=['accuracy'])  
    return model

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
    means = gs_keras.cv_results_['mean_test_score']
    stds = gs_keras.cv_results_['std_test_score']
    params = gs_keras.cv_results_['params']
    data = []
    for mean, stdev, param in zip(means, stds, params):
        data.append([mean, stdev, param])
        results = []
    for param, value in gs_keras.best_params_.items():
        results.append([param, value])
    
    return results, data, gs_keras


X,y = create_data(1000);

# Divide in training / test set
X_mean = X.mean(axis = 0); X_std = X.std(axis = 0)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

### NEURAL NETWORK ###
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

earlystop = EarlyStopping(monitor = 'val_loss', patience=40, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=15, cooldown = 1, min_lr= 0.000009, verbose = 1)

# Grid search
best_result, results, model = searchgrid(X_train, y_train, X_test, y_test)

np.save('GS_results.npy',results); 

model = build_keras(best_result[0][1],best_result[1][1])

history = model.fit(X_train, y_train,
          batch_size = 128,
          epochs = 200,
          verbose = 1,
          validation_data= (X_test, y_test),
          callbacks = [reduce_lr])

model.save_weights('paramstovols_weights.h5')
model.save('paramstovols.h5')


# Comprobar con modelo aleatorio #
import matplotlib.pyplot as plt
testcase = np.random.randint(0,len(X)) ; Xtestcase = np.ravel(X[-testcase,:]);
testvols = np.ravel(model.predict(X[-testcase,:].reshape(1,-1)))

plt.figure()
plt.subplot(121)
plt.plot(Xtestcase[:11], y[-testcase,:], color = 'blue', marker = 'x', linewidth = 1.0);
plt.plot(Xtestcase[:11],testvols, color = 'green', marker = 'o', linewidth = 1.0);
plt.grid(True); plt.legend(['Approx','ANN']); plt.title('volatility smile')

plt.subplot(122)
plt.plot(Xtestcase[:11], (y[-testcase,:] - testvols), linewidth = 1.0);
plt.grid(True); plt.legend(['error']); plt.title('approximation error')
#plt.gca().set_yticklabels(['{:.2f}%'.format(x*100) for x in plt.gca().get_yticks()]) #vol as %
plt.tight_layout()



