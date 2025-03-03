import taichi as ti
from diffusion_algorithms import SuccessiveOverRelaxation
import numpy as np
from numpy.typing import NDArray
from time import sleep


ti.init()


DT = 1.0
DX = 1
DU = 0.16
DV = 0.08
# F = 0.028
# K = 0.062
F = 0.035
K = 0.060

DEBUG = False


@ti.data_oriented
class GrayScott:
    def __init__(self, size: int) -> None:
        self.scaling_matrix = ti.Matrix([[(DT) / (DX**2), 0.0], [0.0, (DT) / (DX**2)]])
        self.diff_matrix = ti.Matrix([[DU, 0.0], [0.0, DV]])
        self.size = size
        self.layers = 2

        self.concentrations = ti.Vector.field(
            self.layers, dtype=float, shape=(size, size)
        )
        self.previous = ti.Vector.field(self.layers, dtype=float, shape=(size, size))

    def init_concentration(
        self,
        u_concentration: NDArray | None = None,
        v_concentration: NDArray | None = None,
        concentration: NDArray | None = None,
    ):
        if concentration is not None:
            assert concentration.shape == (
                self.size,
                self.size,
                2,
            ), f"concentration should be shape {(self.size, self.size, 2)}."
            self._initialize_combined(concentration)
        elif u_concentration is not None and v_concentration is not None:
            assert u_concentration.shape == (
                self.size,
                self.size,
            ), f"u_concentration should be shape {(self.size, self.size)}."
            assert v_concentration.shape == (
                self.size,
                self.size,
            ), f"v_concentration should be shape {(self.size, self.size)}."
            self._initialize_seperate(u_concentration, v_concentration)
        else:
            raise ValueError(
                "Please fill in either both u/v_concentration, or concentration."
            )

    @ti.kernel
    def _initialize_seperate(
        self,
        u_concentration: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
        v_concentration: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
    ):
        for i, j in self.concentrations:
            self.concentrations[i, j][0] = u_concentration[i, j]
            self.concentrations[i, j][1] = v_concentration[i, j]

    @ti.kernel
    def _initialize_combined(
        self, concentration: ti.types.ndarray(dtype=ti.f32, ndim=3)  # type: ignore
    ):
        for i, j in self.concentrations:
            self.concentrations[i, j][0] = concentration[i, j, 0]
            self.concentrations[i, j][1] = concentration[i, j, 1]

    @ti.func
    def bc(self, i: int):
        """
        Periodic boundary conditions.
        """
        return i % self.size

    @ti.func
    def copy_into_previous(self):
        for i, j in self.concentrations:
            self.previous[i, j] = self.concentrations[i, j]

    @ti.func
    def diffuse(self, i: int, j: int):
        # aliases to make formula a (little) bit clearer
        bc, c = ti.static(self.bc, self.previous)
        cij = c[i, j]

        neighbours = self.diff_matrix @ (
            c[bc(i + 1), bc(j)]
            + c[bc(i - 1), bc(j)]
            + c[bc(i), bc(j + 1)]
            + c[bc(i), bc(j - 1)]
            - 4 * c[i, j]
        )

        neighbours += ti.Vector([-(cij[0] * cij[1] ** 2), (cij[0] * cij[1] ** 2)])
        neighbours += ti.Vector([F * (1 - cij[0]), -(F + K) * cij[1]])

        if ti.static(DEBUG):
            print(
                cij,
                ti.Vector([-(cij[0] * cij[1] ** 2), (cij[0] * cij[1] ** 2)]),
                ti.Vector([F * (1 - cij[0]), -(F + K) * cij[1]]),
                neighbours,
            )

        return self.scaling_matrix @ neighbours

    @ti.kernel
    def step_diffusion(self):
        self.copy_into_previous()

        for i, j in self.concentrations:
            self.concentrations[i, j] += self.diffuse(i, j)
        if ti.static(DEBUG):
            print()

    @ti.kernel
    def try_diffuse(self):
        self.diffuse(0, 0)

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.image:
            self.image[i, j][0] = ti.min(
                self.concentrations[i // scale, j // scale][0] * 256, 255
            )
            self.image[i, j][1] = 0
            self.image[i, j][2] = ti.min(
                self.concentrations[i // scale, j // scale][1] * 256, 255
            )

    def gui_loop(self, scale: int = 1, speed: int = 1):
        scaled_size = self.size * scale
        gui = ti.GUI("Gray-Scott Reaction-Diffusion", res=scaled_size)
        self.image = ti.Vector.field(3, ti.u8, shape=(scaled_size, scaled_size))

        while gui.running:
            for _ in range(speed):
                self.step_diffusion()
            self.draw(scale)
            gui.set_image(self.image)
            gui.show()
            # sleep(0.1)
        print(self.concentrations.to_numpy()[2, 2])
        print(self.concentrations.to_numpy()[50, 50])


if __name__ == "__main__":
    N = 100
    gray_scott = GrayScott(N)
    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = 0.25
    # v_concentration[2, 2] = 1.25
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=4, speed=10)
    # gray_scott.step_diffusion()
    # gray_scott.step_diffusion()
    # gray_scott.step_diffusion()
