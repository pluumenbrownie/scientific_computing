import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.animation as animation
import taichi as ti

ti.init(arch = ti.cpu) # change this if you have gpu

# Parameters
D = 1.0  # diffusion coefficient
N = 100   # gridpoints
dx = 1.0 / N  # gridspacing
dt = 0.25 * dx**2 / D  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times

# Initialize grid
c = ti.field(float, shape=(N+1, N+1))
c_new = ti.field(float, shape=(N+1, N+1))

@ti.kernel
def init_concentration():
    for i, j in ti.ndrange(N+1, N+1):  
        if j == 0:  # boundary condition
            c[i, j] = 0  
        elif j == N:  # boundary condition
            c[i, j] = 1  
        else:  
            c[i, j] = 0 


@ti.kernel
def update_concentration():
    for i, j in ti.ndrange((1, N), (1, N)):  # exclude boundaries
        c_new[i, j] = c[i, j] + (dt * D / dx**2) * (
            c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
    
    # apply boundary conditions
    for j in range(N+1):
        c_new[0, j] = 0 
        c_new[N, j] = 1

    # Swap the fields for next iteration
    for i, j in ti.ndrange(N+1, N+1):
        c[i, j] = c_new[i, j]

init_concentration()

# Run the simulation and write data every 100 steps
c_results = []
for step in range(num_steps):
    update_concentration()
    if step % 100 == 0: 
        c_results.append(np.copy(c.to_numpy()))


### there is something wrong here but i cannot see it
def analytical_solution(x, t, D):
    if t == 0:
        return np.zeros_like(x)
    sum_terms = np.zeros_like(x)
    for i in range(50): 
        sum_terms += erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
    return sum_terms

# Write data to a file
np.savetxt("local/numerical_results.txt", c_results[-1])

# E: Plot comparison with analytical solution
y_values = np.linspace(0, 1, N+1)
fig, ax = plt.subplots()
for idx, t in enumerate(times[1:]):
    time_idx = min(int(t / dt) // 100, len(c_results) - 1)
    analytical = analytical_solution(y_values, t, D)
    numerical = c_results[time_idx][:, N//2]  # compare along middle column
    ax.plot(y_values, analytical, '--', label=f'Analytical t={t}')
    ax.plot(y_values, numerical, 'o', label=f'Numerical t={t}')
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
