import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# conjunto de treinamento
x_train = np.array([1.0, 2.0]) # entradas de recursos
y_train = np.array([300.0, 500.0]) # valores de saídas

# calcular função de custo
def calcular_custo(x, y, w, b):
    '''
        Args:
            x - array 
            y - array
    '''
    # m = número de treinamentos
    m = x.shape[0]
    custo = 0
    for i in range(m):
        fwb = w * x[i] + b
        custo = custo + (fwb - y)**2 # somatório de 0 até m
    custo_total = 1/(2*m) * custo
    return custo_total

def calcular_gradiente(x, y, w, b):
    '''
    Args:
        x, y - array numpy
        w, b - escalares parametros
    Return:
        dj_dw, dj_db - derivada parcial dos parâmetros
    '''
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]
    for i in range(m):
        fwb = w * x[i] + b # função linear
        # derivada parcial em relação w: 1/m *(somatorio(fwb - y)*x) 
        # derivada parcial em relação b: 1/m *(somatorio(fwb - y))
        dj_dw +=  (fwb - y)*x[i]
        dj_db += (fwb - y)
    dj_dw /= m # dj_dw / m
    dj_db /= m # dj_db / m
    return dj_dw, dj_db

plt_gradients(x_train, y_train, calcular_custo, calcular_gradiente)
plt.show()

def descida_gradiende(x, y, w_in, b_in, alfa, num_interacoes, funcao_custo, funcao_gradiente):
    '''
        Decida progressiva conforme os o coeficiente de aprendizagem (alpha) e o 
        gradiende (derivada parcial dos parâmetros)
        Args:
            x, y = numpy.array
            w_in, b_in = valores iniciais do modelo
            alfa = coeficiente de aprendizagem, tipo float
            num_interacoes = numero de execuções da decida gradiente
            funcao_custo = produz o custo
            funcao_gradiende = produz o gradiente
        Returns:
            w, b = valores escalares atualizados após a execução do gradiente
            j_history = historico dos valores de custo
            p_history = historico dos parametros [w, b]
    '''
    w = copy.deepcopy(w_in) # evite modificar global
    j_historico = []
    p_historico = []
    w = w_in
    b = b_in

    for i in range(num_interacoes):
        # calcular os gradientes de custo
        dj_dw, dj_db = funcao_gradiende(x, y, w, b)
        # atualizar os parâmetros
        w = w - alfa * dj_dw
        b = b - alfa * dj_db
        if i < 100000:
            j_historico.append(funcao_custo(x, y, w, b))
            p_historico.append([w,b])
        # print os 10 primeiros valores das interações
        if i% math.ceil(num_interacoes/10) == 0:
            print(f"Interação {i:4}: Custo {j_historico[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, j_historico, p_historico # sera usado para montar o grafico    

# parâmetros inicializadores
w_init = 0
b_init = 0

# configurações aleatórias da decida de gradiente
numero_de_interacoes = 10000
coeficiente_de_aprendizagem = 1.0e-2

# executar decida
w_final, b_final, j_hist, p_hist = descida_gradiente(x_train, y_train, w_init, b_init, 
                                    numero_de_interacoes, coeficiente_de_aprendizagem,
                                    calcular_custo, calcular_gradiende)
print(f"(w, b) encontrado por decida gradiente: ({w_final:8.4f}, {b_final:8.4f})")
'''

Iteração    0: Custo 7.93e+04  dj_dw: -6.500e+02, dj_db: -4.000e+02   w:  6.500e+00, b: 4.00000e+00
Iteração 1000: Custo 3.41e+00  dj_dw: -3.712e-01, dj_db:  6.007e-01   w:  1.949e+02, b: 1.08228e+02
...       ...          ...         ...         ...         ...         ...         ...         ...    
Iteração 8000: Custo 1.25e-04  dj_dw: -2.245e-03, dj_db:  3.632e-03   w:  2.000e+02, b: 1.00050e+02
Iteração 9000: Custo 2.90e-05  dj_dw: -1.082e-03, dj_db:  1.751e-03   w:  2.000e+02, b: 1.00024e+02
(w,b) encontrado por decida gradiente: (199.9929,100.0116)

'''

# plot custo vs iterações  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(j_hist[:100])
ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
ax1.set_title("custo vs. interação (inicio)");  ax2.set_title("custo vs. interação (fim)")
ax1.set_ylabel('custo')            ;  ax2.set_ylabel('custo') 
ax1.set_xlabel('interação final')  ;  ax2.set_xlabel('interação final') 
plt.show()

# predições

print(f"1000 m² de casa se prevê {w_final*1.0 + b_final:0.1f} mil dólares")
print(f"1200 m² de casa se prevên {w_final*1.2 + b_final:0.1f} mil dólares")
print(f"2000 m² de casa se prevê {w_final*2.0 + b_final:0.1f} mil dólares")

# plotando o progresso de decida de gradiente numa elipse

fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)

# dando um outro setup para averiguar o progresso
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)

# melhorando o coeficiente de aprendizagem
# parâmetros iniciais
w_init = 0
b_init = 0
# colocando um valor maior para coeficiente de aprendizagem
iterations = 10
tmp_alpha = 8.0e-1
# executar gradiente
w_final, b_final, J_hist, p_hist = descida_gradiente(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, calcular_custo, calcular_gradiende)
'''
Interação    0: Custo 2.58e+05  dj_dw: -6.500e+02, dj_db: -4.000e+02   w:  5.200e+02, b: 3.20000e+02
Interação    1: Custo 7.82e+05  dj_dw:  1.130e+03, dj_db:  7.000e+02   w: -3.840e+02, b:-2.40000e+02
...       ...          ...         ...         ...         ...         ...         ...         ...
Interação    8: Custo 1.85e+09  dj_dw: -5.496e+04, dj_db: -3.397e+04   w:  2.813e+04, b: 1.73730e+04
Interação    9: Custo 5.60e+09  dj_dw:  9.572e+04, dj_db:  5.916e+04   w: -4.845e+04, b:-2.99567e+04
'''
# Acima, 𝑤 e 𝑏 estão oscilando entre positivo e negativo com o valor absoluto aumentando a 
# cada iteração
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
