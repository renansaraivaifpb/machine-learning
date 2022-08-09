import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from modulo import quais dados? # modulo responsavel por retorno de x_train e y_train
from lab_utils_common import dlc # modulo responsavel dos calculos de custo, gradiente e descida de gradiente
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# carregamento do conjunto de dados
X_train, y_train = quais dados?) 
X_features = ['velocidade','profundidade','avanço'] # 3 features(recursos) de cavaco
print(X_train)

# normatilização z-score tornando mais proporcional
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# criacao e ajuste do modelo (alfa, iterações e etc...) | descida gradiente
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# parametros computados
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# fazer predicao com o dominio do conjunto treinamento com sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# a partir dos coeficientes angular e linear w,b. 
# w[i] * x[i] + B (produto escalar entre os vetores w e x usando np.dot (maxima e muito a eficiencia do algoritmo)
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")
print(f"Prediction on training set:\n{y_pred[:10]}" )
print(f"Target values \n{y_train[:10]}")

#Plot Results
# plot predictions and targets vs original features 
# plota as predições vs valores rotulados dos recursos 
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price") 
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
