import numpy as np
import matplotlib.pyplot as plt
from random import *
# predizer preço de casa a partir do seu tamanho

# 1º casa tem 1000 m² e vale $300.000 
# 2º casa tem 2000 m² e vale $500.000

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def calcular_funcao_custo(x,y, w, b):
    '''
    Args:
        x = array (m,) treinos
        y = array (m,) target
        w,b parametros escalares 
    Returns:
        retorna a squared erro cost function 
    '''

    # y-hat = f_wb(x) = w * x(i) + b
    # J(w,b) = somatório de i até m {1/2m [(f_wb - y)]^2}
    # m = numero de treinamento
    m = x.shape[0] # shape é um tuple, se for uma array, sera (2,) se for matriz seria (2,2)
    somatorio_custo = 0
    for i in range(m):
        f_wb = w * x[i] + b
        erro_quadratico = (f_wb - y[i]) ** 2
        somatorio_custo = somatorio_custo + erro_quadratico
    total_custo = (1/(2*m)) * somatorio_custo
    return total_custo
custo = 1
# verificar se consegue obter um minimo a partir dos x(i),y(i)
while(custo != 0.0):
    w = randint(150, 250)
    b = randint(50, 150)
    custo = calcular_funcao_custo(x_train, y_train, w, b)
    print(f"parametro w: {w} e b: {b} gera um custo: {custo}")
print('minimizado')
