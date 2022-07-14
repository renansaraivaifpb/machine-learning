# importação das bibliotecas principais para calculos algebricos


# documentação da biblioteca: https://numpy.org/doc/stable/ 
# https://numpy.org/doc/stable/user/basics.broadcasting.html
import numpy as np 
import time

a = np.zeros(4) # aloca memoria e memoriza arrays
print(f"np.zeros = {a};  a.shape = {a.shape} ;   a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros = {a};  a.shape = {a.shape};   a data type = {a.dtype}")
a = np.random.random_sample(4)
print(f"random_sample = {a};  a.shape = {a.shape};   a data type  = {a.dtype}")

a = np.arange(4)
print(f"np.arange = {a} a shape = {a.shape} a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand = {a}  a shape = {a.shape} a data type = {a.dtype}")

