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


@ti.dataclass
class UV:
    u: float
    v: float


@ti.data_oriented
class GrayScott:
    def __init__(self, size: int) -> None:
        self.diff_const = UV((DT * DU) / (DX**2), (DT * DV) / (DX**2))
        self.size = size

        self.concentrations = UV.field(shape=(size, size))

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
        u_concentration: ti.types.ndarray(dtype=ti.f32, ndim=2),
        v_concentration: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in self.concentrations:
            self.concentrations[i, j].u = u_concentration[i, j]
            self.concentrations[i, j].v = v_concentration[i, j]

    @ti.kernel
    def _initialize_combined(
        self,
        concentration: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for i, j in self.concentrations:
            self.concentrations[i, j].u = concentration[i, j, 0]
            self.concentrations[i, j].v = concentration[i, j, 1]

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.display:
            self.display[i, j].r = ti.min(
                self.concentrations[i // scale, j // scale].u * 256, 255
            )
            self.display[i, j].g = ti.min(
                self.concentrations[i // scale, j // scale].v * 256, 255
            )
            self.display[i, j].b = 0

    def gui_loop(self, scale: int = 1):
        scaled_size = self.size * scale
        gui = ti.GUI("Gray-Scott Reaction-Diffusion", res=(scaled_size, scaled_size))  # type: ignore
        self.display = ti.Vector.field(3, ti.u8, shape=(scaled_size, scaled_size))

        while gui.running:
            self.draw(scale)
            gui.set_image(self.display)
            gui.show()


if __name__ == "__main__":
    N = 100
    gray_scott = GrayScott(N)
    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 2 - 5 : N // 2 + 5, N // 2 - 5 : N // 2 + 5] = 0.25
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=4)
