import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def func(x,y):
    return ((x -x**3/6 + x**5/120) + (1 - y**2/2 + y**4/24))

def grad_partial(x, y):
    dfdx = 1 - x**2/2 + x**4/24
    dfdy = -y + y**3/6

    return np.array([dfdx, dfdy])

tolerance = 1e-5
max_steps = 500
alpha = 0.1
x0 = np.array([1.0, 1.0])
path = [x0.copy()]
x = x0.copy()

for i in range(max_steps):
    grad = grad_partial(x[0], x[1])
    if np.linalg.norm(grad) < tolerance:
        print(f"Convergencia alcanzada en paso {i}")
        break
    x -= alpha * grad
    path.append(x.copy())

path = np.array(path)


min_point = path[-1]
min_value = func(min_point[0], min_point[1])
print(f"El punto mínimo hallado es aproximadamente: (x, y) = ({min_point[0]:.4f}, {min_point[1]:.4f})")
print(f"El valor de la función en ese punto es: f(x, y) = {min_value:.4f}")

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

ball, = ax.plot([], [], [], 'ro', markersize=8)
traj, = ax.plot([], [], [], 'r--', lw=1)  # Trayectoria

def init():
    ball.set_data([], [])
    ball.set_3d_properties([])
    traj.set_data([], [])
    traj.set_3d_properties([])
    return ball, traj

def update(frame):
    x, y = path[frame]
    z = func(x, y)
    ball.set_data([x], [y])
    ball.set_3d_properties([z])
    traj.set_data(path[:frame+1,0], path[:frame+1,1])
    traj.set_3d_properties([func(xx, yy) for xx,yy in path[:frame+1]])
    return ball, traj

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Descenso por Gradiente ')
fig.colorbar(surf, shrink=0.5, aspect=10)

ani = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, interval=300)
plt.show()
