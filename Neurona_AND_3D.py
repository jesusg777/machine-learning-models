"""
Solución compuerta AND en 3D con Perceptrón
Juan David Yepez Velez - 89783
Jesus David Gelves Cajiao - 98650
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def step_function(x):
    return np.where(x >= 0, 1, 0)

def perceptron_predict(X, weights):
    return step_function(np.dot(X, weights[1:]) + weights[0])

# Algoritmo del Perceptrón
def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    # Inicializar los pesos (uno más para el bias)
    weights = np.random.rand(X.shape[1] + 1)#np.zeros(X.shape[1] + 1)
    #vector de error
    errors = []

    # Entrenamiento
    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Calcular la salida (predicción)
            output = perceptron_predict(xi,weights)#step_function(np.dot(xi, weights[1:]) + weights[0])
            # Calcular error absoluto
            error = target - output
            total_error += abs(error)
            # Actualizar los pesos
            update = learning_rate * (target - output)
            weights[1:] += update * xi
            weights[0] += update
        errors.append(total_error)
    return weights,errors

# Preparar los datos de entrada y salida
# Datos de entrada para la compuerta AND
X=np.array([[0,0,0],
      [0,0,1],
      [0,1,0],
      [0,1,1], 
      [1,0,0],
      [1,0,1],  
      [1,1,0],
      [1,1,1]])
# Salidas esperadas para la compuerta AND
y = np.array([0, 0, 0, 0, 0, 0, 0, 1])
# Salidas esperadas para la compuerta OR
#y = np.array([0, 1, 1, 1])

# Entrenar el perceptrón
weights,errors = perceptron_train(X, y, learning_rate=0.1, epochs=10)
print("Pesos entrenados:", weights)

# Graficar el error global en cada época
plt.figure(1)
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Época')
plt.ylabel('Error Global')
plt.title('Error Global del Perceptrón en cada Época')
plt.grid(True)

# Probar el perceptrón con los datos de entrada
for xi in X:
    prediction = perceptron_predict(xi, weights)
    print(f"Entrada: {xi}, Salida predicha: {prediction}")


# Visualizacion en 3D
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

# Graficar puntos
for i, xi in enumerate(X):
    if y[i] == 0:
        ax.scatter(xi[0], xi[1], xi[2], color='red', marker='o', label='Clase 0' if i == 0 else "")
    else:
        ax.scatter(xi[0], xi[1], xi[2], color='blue', marker='^', label='Clase 1' if i == 7 else "")

# w0 + w1*x + w2*y + w3*z = 0 -> z = (-w0 - w1*x - w2*y) / w3
w0, w1, w2, w3 = weights
x_vals = np.linspace(0, 1, 10)
y_vals = np.linspace(0, 1, 10)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_grid = -(w0 + w1 * X_grid + w2 * Y_grid) / w3

ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5, color='green')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Perceptrón AND - Plano de decisión')
ax.legend()
plt.tight_layout()
plt.show()
