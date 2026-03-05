"""
Solución tabla de representacion binaria
Juan David Yepez Velez - 89783
Jesus David Gelves Cajiao - 98650
"""

import numpy as np
import matplotlib.pyplot as plt


# matriz 5x3, numeros del 0 al 9
# cada fila representa un dígito en formato binario
digitos = {
    0: [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    1: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    2: [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    3: [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    4: [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    5: [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    6: [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    7: [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    8: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
}

X = np.array([digitos[i] for i in range(10)]) #Entrada binarai
Y = np.array([[int(b) for b in format(i, '04b')] for i in range(10)])  # salidas binarios de 4 bits

# Función de activación
def step(x):
    return np.where(x >= 0, 1, 0)

# Entrenamiento del perceptrón por bit
def train_perceptron(X, y, epochs=100, lr=0.1):
    weights = np.random.rand(X.shape[1] + 1)
    for _ in range(epochs):
        for xi, target in zip(X, y):
            xi_bias = np.insert(xi, 0, 1)  # agregar bias
            output = step(np.dot(weights, xi_bias))
            error = target - output
            weights += lr * error * xi_bias
    return weights

# Entrenar 4 neuronas (una por bit de salida)
all_weights = []
for bit_index in range(4):
    y_bit = Y[:, bit_index]
    w = train_perceptron(X, y_bit)
    all_weights.append(w)

# Función de predicción
def predict(xi, weights_list):
    xi_bias = np.insert(xi, 0, 1)
    return [step(np.dot(w, xi_bias)) for w in weights_list]

# Probar predicciones
for i, xi in enumerate(X):
    prediction = predict(xi, all_weights)
    print(f"Entrada: {i} → Salida esperada: {Y[i]} | Predicción: {prediction}")

def ruido_aleatorio(x, noise_level=0.1):
    #Añadir ruido aleatorio a los datos de entrada.
    x= np.array(x)
    ruido= np.random.rand(*x.shape) * noise_level
    x_ruido= np.where(ruido,1 -x,x)
    return x_ruido

digito_original = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]  # el número 0
digito_con_ruido = ruido_aleatorio(digito_original, noise_level=0.2)

print("Original:", digito_original)
print("Con ruido:", digito_con_ruido.tolist())

def graficar_lado_a_lado(original, ruidoso, titulo1="Original", titulo2="Con Ruido"):
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    # Convierte a matriz 5x3
    original = np.array(original).reshape(5, 3)
    ruidoso = np.array(ruidoso).reshape(5, 3)

    for ax, mat, title in zip(axes, [original, ruidoso], [titulo1, titulo2]):
        ax.imshow(mat, cmap='Greys', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Ejemplo con el 0
graficar_lado_a_lado(digito_original, digito_con_ruido, "Dígito 0 Original", "Dígito 0 con Ruido")
