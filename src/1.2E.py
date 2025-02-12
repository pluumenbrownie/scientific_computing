import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.animation as animation

# Parameters
D = 1.0  # diffusion coefficient
N = 100   # gridpoints
dx = 1.0 / N  # gridspacing
dt = 0.25 * dx**2 / D  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times

# Initialize grid
c = np.zeros((N+1, N+1))  # the boundaries
c[:, -1] = 1  # top boundary
c[:, 0] = 0   # bottom boundary

def update_concentration(c):
    c_new = np.copy(c)
    for i in range(1, N):  # exclude boundaries
        for j in range(1, N):
            c_new[i, j] = c[i, j] + (dt * D / dx**2) * (
                c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4*c[i, j])
    
    # Apply boundary conditions
    c_new[:, 0] = 0   # bottom boundary
    c_new[:, -1] = 1  # top boundary
    return c_new

# Run the simulation and write data every 100 steps
c_results = []
for step in range(num_steps):
    c = update_concentration(c)
    if step % 100 == 0:
        c_results.append(np.copy(c))

### there is something wrong here but i cannot see it
def analytical_solution(x, t, D):
    if t == 0:
        return np.zeros_like(x)
    sum_terms = np.zeros_like(x)
    for i in range(50): 
        sum_terms += erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
    return sum_terms

# Write data to a file
np.savetxt("numerical_results.txt", c_results[-1])

# E: Plot comparison with analytical solution
x_values = np.linspace(0, 1, N+1)
fig, ax = plt.subplots()
for idx, t in enumerate(times[1:]):
    analytical = analytical_solution(x_values, t, D)
    numerical = c_results[min(idx, len(c_results)-1)][:, N//2]  # compare along middle column
    ax.plot(x_values, analytical, '--', label=f'Analytical t={t}')
    ax.plot(x_values, numerical, 'o', label=f'Numerical t={t}')
ax.set_xlabel('y')
ax.set_ylabel('c(y)')
ax.legend()
ax.set_title("Comparison of numerical and analytical solutions")
plt.show()

# F: Plot 2D concentration map at different times
fig, axes = plt.subplots(1, len(times), figsize=(15, 3))
for idx, t in enumerate(times):
    im = axes[idx].imshow(c_results[min(idx, len(c_results)-1)], cmap='hot', origin='lower', extent=[0,1,0,1])
    axes[idx].set_title(f't = {t}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05)
plt.show()

# Part G: Animation of the diffusion
fig, ax = plt.subplots()
cmap = ax.imshow(c_results[0], cmap='hot', origin='lower', extent=[0,1,0,1])
ax.set_title("Animation of diffusion")

def animate(i):
    cmap.set_array(c_results[i])
    ax.set_title(f'Time step {i*100}')
    return [cmap]

ani = animation.FuncAnimation(fig, animate, frames=len(c_results), interval=100, blit=False)
plt.show()
