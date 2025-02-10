import taichi as ti 
import taichi.math as tm


ti.init()


# init boilerplate
N: int = 1_000
dx: float = 1/N
dt = 0.001
c: int = 1
amplitude = ti.field(float, shape = N)  # u[i,j]
amplitude_previous = ti.field(float, shape = N)  # u[i,j-1]
amplitude_next = ti.field(float, shape=N) # u[i,j+1]


@ti.kernel
def init_string_1():
    """
    Psi(x, t=0) = sin(2 pi x)
    """
    for i in range(N):
        amplitude[i] = tm.sin(2 * tm.pi * i * dx)
        amplitude_previous[i] = tm.sin(2 * tm.pi * i * dx)


@ti.kernel
def init_string_2():
    """
    Psi(x, t=0) = sin(5 pi x)
    """
    for i in range(N):
        amplitude[i] = tm.sin(5 * tm.pi * i * dx)
        amplitude_previous[i] = tm.sin(5 * tm.pi * i * dx)


@ti.kernel
def init_string_3():
    """
    Psi(x, t=0) = sin(5 pi x) if 1/5 < x < 2/5
    """
    for i in range(N):
        if i * dx > 1/5.0 and i*dx < 2/5:
            amplitude[i] = tm.sin(5 * tm.pi * i * dx)
            amplitude_previous[i] = tm.sin(5 * tm.pi * i * dx)
        else:
            amplitude[i] = tm.sin(0.0)
            amplitude_previous[i] = tm.sin(0.0)


@ti.kernel
def update_string():
    """ 
    update the string in one time step
    """
    constant = (c**2) * (dt**2) / (dx**2)
    amplitude_next[0] = 0 # boundary condition
    amplitude_next[N-1] = 0 # boundary condition
    for i in range(1,N-1): 
        change = amplitude[i+1] + amplitude[i-1] - 2 * amplitude[i]
        amplitude_next[i] = 2 * amplitude[i] - amplitude_previous[i] + constant * change
    
    for i in range(N): # update the elements in the fields
        amplitude_previous[i] = amplitude[i]
        amplitude[i] = amplitude_next[i]
