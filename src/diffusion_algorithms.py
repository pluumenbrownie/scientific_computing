from typing import Callable
import numpy as np
import taichi as ti
import math as mt


ti.init(arch=ti.cpu)  # change this if you have gpu


# Parameters
D = 1.0  # diffusion coefficient
N = 50  # gridpoints
DEFAULT_N = 50  # gridpoints
OMEGA = 1.8  # relaxation constant
DEFAULT_OMEGA = 1.8  # relaxation constant
THRESHOLD = 1e-5
dx = 1.0 / N  # gridspacing
dt = 0.25 * dx**2 / D  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times

image_scale = 4

# Initialize grid
concentration = ti.field(float, shape=(N, N))
c_difference = ti.field(float, shape=(N, N))
white_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.ceil(N**2 / 2)))
black_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.floor(N**2 / 2)))


@ti.data_oriented
class BaseIteration:
    def __init__(self, N: int = DEFAULT_N, threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.N = N
        self.concentration = ti.field(float, shape=(N, N))
        self.c_difference = ti.field(float, shape=(N, N))
        self.init()
    
    def run(self) -> int:
        self.solve()
        runs = 1
        while self.c_difference.to_numpy().max() > self.threshold:
            self.solve()
            runs += 1
        return runs
    
    def init(self):
        self.init_concentration()

    @ti.kernel 
    def solve(self):
        pass

    @ti.func
    def pbi(self, i: int) -> int:
        """
        Convert normal coordinates into periodic boundary coordinates.
        """
        if i < 0:
            i += self.N
        if i >= self.N:
            i -= self.N
        return i
        
    @ti.func
    def neighbourhood_values(self, i: int, j: int, field) -> float:
        """
        Get the combined values of the four neighbours of the given cell,
        using boundary conditions.
        """
        return (
            field[self.pbi(i - 1), j]
            + field[self.pbi(i + 1), j]
            + field[self.pbi(i), j - 1]
            + field[self.pbi(i), j + 1]
        )

    @ti.kernel
    def init_concentration(self):
        for i, j in self.concentration:
            if j == 0:  # boundary condition
                self.concentration[i, j] = 0
            elif j == self.N - 1:  # boundary condition
                self.concentration[i, j] = 1
            else:
                self.concentration[i, j] = 0

    @ti.func
    def copy_into_difference(self):
        for i, j in self.concentration:
            self.c_difference[i, j] = self.concentration[i, j]
    
    @ti.func
    def calculate_differences(self):
        for i, j in self.c_difference:
            self.c_difference[i, j] = abs(self.concentration[i, j] - self.c_difference[i, j])
                
    @ti.func
    def reset_boundary(self):
        for i in range(self.N):
            self.concentration[i, 0] = 1.0
            self.concentration[i, self.N - 1] = 0.0

    def gui(self, scale: int = 10):
        gui = ti.GUI(f"Diffusion {self.N}x{self.N}", res=(scale * self.N, scale * self.N))  # type:ignore
        image = self.concentration.to_numpy()
        if not scale == 1:
            scaled_image = np.zeros(shape=(scale * self.N, scale * self.N))
            for i, j in np.ndindex(scaled_image.shape):
                scaled_image[i, j] = image[i // scale, j // scale]
            image = scaled_image
        while gui.running:
            gui.set_image(image)
            gui.show()


class Jacobi(BaseIteration):
    @ti.kernel
    def solve(self):
        self.copy_into_difference()
        for i, j in self.concentration:
            if j == 0 or j >= N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(i, j, self.c_difference)
        self.reset_boundary()
        self.calculate_differences()


class GaussSeidel(BaseIteration):
    def __init__(self, N: int = N, threshold: float = THRESHOLD) -> None:
        super().__init__(N, threshold)
        self.white_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.ceil(N**2 / 2)))
        self.black_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.floor(N**2 / 2)))
        self.init_checkerboard()
    
    @ti.kernel
    def init_checkerboard(self):
        for i, j in self.concentration:
            if i % 2 == j % 2:
                self.white_tiles[j // 2 + ti.ceil(i * self.N / 2, dtype=int)] = ti.Vector([i, j])
            else:
                self.black_tiles[j // 2 + ti.floor(i * self.N / 2, dtype=int)] = ti.Vector([i, j])
    
    @ti.kernel
    def solve(self):
        self.copy_into_difference()
        for v in self.black_tiles:
            i, j = int(self.black_tiles[v][0]), int(self.black_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(i, j, self.concentration)

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(i, j, self.concentration)
        self.calculate_differences()


class SuccessiveOverRelaxation(GaussSeidel):
    def __init__(self, omega: float = DEFAULT_OMEGA, N: int = N, threshold: float = THRESHOLD) -> None:
        super().__init__(N, threshold)
        self.omega = omega
    

    @ti.kernel 
    def solve(self):
        self.copy_into_difference()
        for v in self.black_tiles:
            i, j = int(self.black_tiles[v][0]), int(self.black_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = (
                self.omega * 0.25 * self.neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j]
            )

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = (
                self.omega * 0.25 * self.neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j]
            )
        self.calculate_differences()

if __name__ == "__main__":
    # jacobi = Jacobi()
    # class_amount = jacobi.run()
    # gauss = GaussSeidel()
    # gauss_amount = gauss.run()
    # print(f"{gauss_amount = }")
    # gauss.gui()
    sov = SuccessiveOverRelaxation()
    sov_amount = sov.run()
    print(f"{sov_amount = }")
    sov.gui()
