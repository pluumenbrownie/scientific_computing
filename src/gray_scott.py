import taichi as ti
from diffusion_algorithms import SuccessiveOverRelaxation
import numpy as np
from numpy.typing import NDArray


ti.init()


DT = 1.0
DX = 1
DU = 0.16
DV = 0.08
F = 0.035
K = 0.060


@ti.data_oriented
class GrayScott:
    def __init__(self, size: int) -> None:
        self.diff_const = ti.types.vector([(DT * DU) / (DX**2), (DT * DV) / (DX**2)])
        self.size = size

        self.concentrations = ti.Vector.field(size, dtype=float, shape=(size, size))
        self.previous = ti.Vector.field(size, dtype=float, shape=(size, size))

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
        print(u_concentration[0, 0])

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
        neighbours = (
            c[bc(i + 1), bc(j)]
            + c[bc(i - 1), bc(j)]
            + c[bc(i), bc(j + 1)]
            + c[bc(i), bc(j - 1)]
            - 4 * c[i, j]
        )
        # neighbours.outer_product(self.diff_const)
        # neighbours[0] *= self.diff_const[0]
        # neighbours[1] *= self.diff_const[1]
        # return neighbours

    @ti.kernel
    def step_diffusion(self):
        self.copy_into_previous()

        for i, j in self.concentrations:
            self.diffuse(i, j)
            # self.concentrations[i, j] += self.diffuse(i, j)
            # self.concentrations[i, j][1] = self.diffuse(i, j, 1)

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.display:
            self.display[i, j][0] = ti.min(
                self.concentrations[i // scale, j // scale][0] * 256, 255
            )
            self.display[i, j][1] = ti.min(
                self.concentrations[i // scale, j // scale][1] * 256, 255
            )
            self.display[i, j][2] = 0

    def gui_loop(self, scale: int = 1):
        scaled_size = self.size * scale
        gui = ti.GUI("Gray-Scott Reaction-Diffusion", res=scaled_size)
        self.display = ti.Vector.field(3, ti.u8, shape=(scaled_size, scaled_size))

        while gui.running:
            self.step_diffusion()
            self.draw(scale)
            gui.set_image(self.display)
            gui.show()


if __name__ == "__main__":
    N = 5
    gray_scott = GrayScott(N)
    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = 0.25
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=40)
