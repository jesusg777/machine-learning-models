"""
Perceptrón 3D con datos en: cubo, esfera o tetraedro
Autor: (tu equipo)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D

np.random.seed(42)

# ------------ Perceptrón (simple) ------------
def step_function(x):
    return np.where(x >= 0, 1, 0)

def perceptron_predict(X, weights):
    # weights[0] = bias, weights[1:] = w
    return step_function(np.dot(X, weights[1:]) + weights[0])

def perceptron_train(X, y, lr=0.1, epochs=30):
    weights = np.zeros(X.shape[1] + 1)   # [b, w1, w2, w3]
    errors = []
    for _ in range(epochs):
        idx = np.random.permutation(len(X)) # indices aleatorios para que el orden no afecte
        err = 0
        for xi, target in zip(X[idx], y[idx]):
            pred = perceptron_predict(xi, weights)  # escalar 0/1
            update = lr * (target - pred)           # escalar
            if update != 0:
                weights[1:] += update * xi
                weights[0]  += update
                err += 1
        errors.append(err)
    return weights, errors


# ------------ Generadores de puntos ------------
def gen_cubo(n=400, low=-1, high=1):
    return np.random.uniform(low, high, size=(n, 3))

def gen_esfera(n=400, r=1.0):
    pts = []
    while len(pts) < n:
        xyz = np.random.uniform(-r, r, size=(n, 3))
        inside = xyz[np.sum(xyz**2, axis=1) <= r**2]
        pts.extend(list(inside))
    return np.array(pts[:n])

def gen_tetraedro(n=400):
    V = np.array([[ 1, 1, 1],
                  [ 1,-1,-1],
                  [-1, 1,-1],
                  [-1,-1, 1]], dtype=float) / np.sqrt(3)
    alphas = np.random.dirichlet([1,1,1,1], size=n)
    return alphas @ V

# ------------ Etiquetado lineal (plano) ------------
def etiquetar_por_plano(X):
    v = np.random.normal(size=3); v /= (np.linalg.norm(v) + 1e-12)
    b = np.random.uniform(-0.2, 0.2)
    scores = X @ v + b
    y = (scores >= 0).astype(int)
    return y, v, b

# ------------ Elegir figura ------------
figura = "cubo"   # "cubo" | "esfera" | "tetraedro"
if figura == "cubo":
    X = gen_cubo(n=400)
elif figura == "esfera":
    X = gen_esfera(n=400, r=1.0)
elif figura == "tetraedro":
    X = gen_tetraedro(n=400)
else:
    raise ValueError("Figura no válida.")

y, w_true, b_true = etiquetar_por_plano(X)

# ------------ Entrenamiento ------------
weights_learn, errs = perceptron_train(X, y, lr=0.1, epochs=40)

print("Plano REAL:    w=", w_true, " b=", b_true)
print("Plano APREND.: w=", weights_learn[1:], " b=", weights_learn[0])

# Exactitud (sobre el mismo set)
acc = (perceptron_predict(X, weights_learn) == y).mean()
print(f"Exactitud: {acc:.3f}")

# ------------ Gráfica de error ------------
plt.figure(1)
plt.plot(range(1, len(errs)+1), errs, marker='o')
plt.xlabel("Época"); plt.ylabel("Errores")
plt.title(f"Errores por época - {figura.capitalize()}")
plt.grid(True)

# ------------ 3D: puntos + plano aprendido ------------
fig = plt.figure(2); ax = fig.add_subplot(111, projection="3d")
c = np.where(y==0, 'red', 'blue')
ax.scatter(X[:,0], X[:,1], X[:,2], c=c, s=12, alpha=0.7)

# Plano: b + w1*x + w2*y + w3*z = 0  -> z = -(b + w1*x + w2*y)/w3
b_learn = weights_learn[0]
w1, w2, w3 = weights_learn[1:]
xs = np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 20)
ys = np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1, 20)
Xg, Yg = np.meshgrid(xs, ys)
if abs(w3) > 1e-8:
    Zg = -(b_learn + w1*Xg + w2*Yg) / w3
    ax.plot_surface(Xg, Yg, Zg, alpha=0.4)

ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
ax.set_title(f"{figura.capitalize()} - Perceptrón (plano de decisión)")
plt.tight_layout(); plt.show()
