import numpy as np
import time

def loop_array(a, b):
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x
np.random.seed(1)

# arrays mt grandes

a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time() # captura o tempo inicial
c = np.dot(a, b)
tac = time.time() # captura o tempo final
print(f"np.dot(a, b) = {c:.4f}")
print(f"tempo de: {1000*(tac-tic):.4f} ms")

tic = time.time() # captura o tempo inicial
l = loop_array(a, b)
tac = time.time() # captura o tempo final
print(f"loop_array = {l:.4f}")
print(f"tempo de: {1000*(tac-tic):.4f} ms")