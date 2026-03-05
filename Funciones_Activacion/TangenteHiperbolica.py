import numpy as np 
import matplotlib.pyplot as plt

def tanh_hyperbolic(x):
    return np.tanh(x)

x = np.linspace(-2, 2, 100)
y = tanh_hyperbolic(x)

# Graficar
plt.figure(figsize=(6, 4))
plt.plot(x, y, label='Función escalón', color='blue')
plt.title('Función de activación escalón (Heaviside)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
