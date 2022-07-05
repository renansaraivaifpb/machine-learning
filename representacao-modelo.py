from cProfile import label
import numpy as np
# numpy é uma biblioteca de computação científica
import matplotlib.pyplot as plt
# matplotlib é uma biblioteca para plottar dados


# trabalhando primeira predição 
# treinamento de dados para predizer o melhor preço de venda visando apenas uma única característica:
# tamanho 1.000m² = $1.000

# input(entradas) = tamanhos
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


print(f"x_train.shape = {x_train.shape} type = {type(x_train.shape)}")

# m = number of training examples
m = x_train.shape[0] # = 2 or use function len(x_train) = 2
# x1 e x2
# y1 e y2
print(f"number of training examples is {m}")

# cada entrada e saida será colocada em indices, como se fosse uma array
# {x(𝑖) , y(𝑖)}
i = 0
x_i = x_train[i]
y_i = y_train[i]

w = 200
b = 100
print(f"w = {w} e b = {b}")

# como são dois numeros de treinamentos, teremos que encontrar o f w,b (x^(1)) para montarmos a função
# no gráfico
def calcule_saida_modelo(x, w, b):
    '''
    Função que calcula um modelo linear
    x = np.array(m,) - x_train
    w,b = escalares - pesos
    f_wb = função linear
    '''
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb
tmp_f_wb = calcule_saida_modelo(x_train, w, b)
print(f"x({i}) , y({i}) = {x_i} , {y_i}")

def criar_grafico(x_train, y_train, tmp_f_wb):
    # colocar uma linha linear modelo no gráfico x e y-hat 
    plt.plot(x_train, tmp_f_wb, c='b', label='predição calculada')
    # 'marker' define o formato da coordenada, 'c' é para cor
    plt.scatter(x_train, y_train, marker='x', c='r',label='coordenada do valor atual')
    # titulo do gráfico
    plt.title("Predição do valor residencial")
    # rótulo para o eixo x
    plt.xlabel("tamanho da casa")
    # rótulo para o eixo y
    plt.ylabel("valor da casa")
    plt.legend()
    plt.show()

criar_grafico(x_train, y_train, tmp_f_wb)
# e se agora tivemos mais um dado?
# antes de tudo, precisamos colocar nossa entrada e calcular seu custo para plotar no grafico
x_train = np.array([1.0, 1.2, 2.0])
valor_1200 = w*x_train[1] + b
y_train = np.array([300.0,valor_1200, 500.0])
tmp_f_wb = calcule_saida_modelo(x_train, w, b)

criar_grafico(x_train, y_train, tmp_f_wb)
