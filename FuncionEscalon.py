import numpy as np
import matplotlib.pyplot as plt

# Definimos la función escalón
def f(x):
    return np.where(x < 0, 0, 1)

# Dominio de x
x = np.linspace(-2, 2, 100)
y = f(x)

# Graficar
plt.figure(figsize=(6, 4))
plt.plot(x, y, label='Función escalón', color='blue')
plt.title('Función de activación escalón (Heaviside)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
