import numpy as np
import matplotlib.pyplot as plt

# Función de activación (escalón)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Predicción del perceptrón
def perceptron_predict(X, weights):
    return step_function(np.dot(X, weights[1:]) + weights[0])

# Entrenamiento del perceptrón
def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    weights = np.random.rand(X.shape[1] + 1)  # Inicialización aleatoria
    errors = []

    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            output = perceptron_predict(xi, weights)
            error = target - output
            update = learning_rate * error
            weights[1:] += update * xi
            weights[0] += update
            total_error += abs(error)
        errors.append(total_error)

    return weights, errors

# Entradas
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas esperadas
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])

# Entrenar perceptrones
weights_and, errors_and = perceptron_train(X, y_and)
weights_or,  errors_or  = perceptron_train(X, y_or)

# Mostrar predicciones
print("Predicciones con perceptrón para AND y OR:\n")
for xi in X:
    pred_and = perceptron_predict(xi, weights_and)
    pred_or  = perceptron_predict(xi, weights_or)
    print(f"Entrada: {xi} → AND: {pred_and} | OR: {pred_or}")

# Graficar errores
plt.plot(range(1, len(errors_and)+1), errors_and, label='AND', marker='o')
plt.plot(range(1, len(errors_or)+1), errors_or, label='OR', marker='x')
plt.xlabel("Época")
plt.ylabel("Error total")
plt.title("Error por época - AND vs OR")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------
# Gráfica 1: Línea de decisión para compuerta AND
# -----------------------------------------------
plt.figure(figsize=(6, 5))
for i, xi in enumerate(X):
    if y_and[i] == 0:
        plt.scatter(xi[0], xi[1], color='red', marker='o', label='Clase 0 (AND)' if i == 0 else "")
    else:
        plt.scatter(xi[0], xi[1], color='blue', marker='x', label='Clase 1 (AND)' if i == 3 else "")

# Línea de decisión AND
x_vals = np.array([0, 1])
y_vals = -(weights_and[1] * x_vals + weights_and[0]) / weights_and[2]
plt.plot(x_vals, y_vals, color='green', label='Línea de decisión AND')

plt.title("Compuerta AND - Línea de Decisión")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------------------------
# Gráfica 2: Línea de decisión para compuerta OR
# -----------------------------------------------
plt.figure(figsize=(6, 5))
for i, xi in enumerate(X):
    if y_or[i] == 0:
        plt.scatter(xi[0], xi[1], color='red', marker='o', label='Clase 0 (OR)' if i == 0 else "")
    else:
        plt.scatter(xi[0], xi[1], color='blue', marker='x', label='Clase 1 (OR)' if i == 1 else "")

# Línea de decisión OR
y_vals_or = -(weights_or[1] * x_vals + weights_or[0]) / weights_or[2]
plt.plot(x_vals, y_vals_or, color='purple', label='Línea de decisión OR')

plt.title("Compuerta OR - Línea de Decisión")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.show()
