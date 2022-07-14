import numpy as np

def loop_dot(w, x):
    '''
    Args:
        W, X - vetores de mesmas dimensões
    Return:
        Produto escalar entre vetores com loop
        r = w * x 
    '''
    r = 0
    for i in range(x.shape[0]):
        r += w[i] * x[i]
    return r
# teste

w = np.arange(5)
x = np.arange(5)
print(w,x)
print(f"somatório (w * x) = {loop_dot(w, x)}")

# produto escalar entre vetores com thrends
print(f"np.dot = {np.dot(w,x)}")