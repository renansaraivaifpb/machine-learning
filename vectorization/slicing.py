import numpy as np

a = np.arange(10)
print(a),

# [start: end: step]
# mostrar dois em dois 0 até 10
print(f"a[5:9:1] = {a[5:10:1]}")
# acesso todos os elementos pulando 3 em 3
print(f"a[::3] = {a[::3]}")
# acesso todos elementos atras do 5
print(f"a[:5:1] = {a[:5:1]}")
# acesso a todos os elentos
print(f"a[:] = {a[:]}")