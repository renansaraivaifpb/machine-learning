import numpy as np
a = np.array([1, 3, 4, 6])
print(f"a:   {a}")
# todos negativos
b = -a
print(f"a- : {b}")
# somando todos os elementos da array
b = np.sum(a)
print(f"np.sum(a) = {b}")
# calculando a média 
b = np.mean(a)
print(f"np.mean(a) = {b}")
# ao quadrado 
b = a**2
print(f"a**2 = {b}")
# soma entre arrays
vetor1 = np.random.random_sample(5)
vetor2 = np.random.random_sample(5)
vetor3 = np.random.random_sample(4)
print(vetor1)
print(vetor2)
print(vetor3)
try:
    print(f"vetor1 - vetor2 = {vetor1 - vetor2}")
    print(f"vetor2 + vetor3 = {vetor2 + vetor3}")
except Exception as error:
    print(f"Erro is -> {error}")


# multiplicação escalar

a = np.random.rand(5)
print(f"a = {a}")
a = a * 5
print(f"a * 5 = {a}")

# Matrizes
a = np.arange(6).reshape(-1, 2)   # reshape uma alternativa para se criar uma matriz calcula as linhas por: 6/2 = 3
print(f"a.shape: {a.shape}, \na= {a}")

# acessando elemento
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

# acesslando coluna
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
