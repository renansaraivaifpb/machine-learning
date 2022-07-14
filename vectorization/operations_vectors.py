import numpy as np

a = np.arange(10)
print(a)

print(f"a[2].shape = {a[2].shape} | a[5] = {a[5]}")

# acessar o ultimo elemento
print(f"a[-1] = {a[-1]}")

try:
    print(f"a[10] = {a[10]}")
except Exception as e:
    print(f"error -> {e}")