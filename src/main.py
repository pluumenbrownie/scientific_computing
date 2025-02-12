# Question 1.2D
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 5   # number of grid points in each direction

# Visualization of Grid Indices
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(N+1):
    for j in range(N+1):
        ax.scatter(i, j, color='black', s=40)
        ax.text(i, j, f'({i},{j})', fontsize=8, ha='right', color='blue')

ax.set_xlim(-1, N+1)
ax.set_ylim(-1, N+1)
ax.set_xticks(range(N+1))
ax.set_yticks(range(N+1))
ax.set_xticklabels(range(N+1))
ax.set_yticklabels(range(N+1))
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_title("Grid Indices and Boundary Conditions")
plt.gca().invert_yaxis()
plt.show()
