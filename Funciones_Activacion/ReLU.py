import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Dominio de x
x = np.linspace(-2, 2, 100)
y = relu(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y, label='Función escalón', color='blue')
plt.title('Función de activación escalón (Heaviside)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()