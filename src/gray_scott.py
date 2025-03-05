import taichi as ti
import numpy as np
from numpy.typing import NDArray


ti.init(arch=ti.gpu)


SPEED = 1000


@ti.data_oriented
class GrayScott:
    def __init__(
        self,
        size: int,
        DT: float = 1.0,
        DX: float = 1.0,
        DU: float = 0.16,
        DV: float = 0.08,
        F: float = 0.035,
        K: float = 0.060,
    ) -> None:
        self.scaling_matrix = ti.Matrix([[(DT) / (DX**2), 0.0], [0.0, (DT) / (DX**2)]])
        self.diffusion_constants = ti.Matrix([[DU, 0.0], [0.0, DV]])
        self.size = size
        self.F = F
        self.K = K
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

        neighbours = self.diffusion_constants @ (
            c[bc(i + 1), bc(j)]
            + c[bc(i - 1), bc(j)]
            + c[bc(i), bc(j + 1)]
            + c[bc(i), bc(j - 1)]
            - 4 * cij
        )

        neighbours += ti.Vector([-(cij[0] * cij[1] ** 2), (cij[0] * cij[1] ** 2)])
        neighbours += ti.Vector([self.F * (1 - cij[0]), -(self.F + self.K) * cij[1]])

        return self.scaling_matrix @ neighbours

    @ti.kernel
    def step_diffusion(self):
        self.copy_into_previous()

        for i, j in self.concentrations:
            self.concentrations[i, j] += self.diffuse(i, j)

    @ti.kernel
    def try_diffuse(self):
        self.diffuse(0, 0)

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.image:
            self.image[i, j][0] = self.concentrations[i // scale, j // scale][
                0
            ] - ti.max(self.concentrations[i // scale, j // scale][1] - 0.3, 0.0)
            self.image[i, j][1] = self.concentrations[i // scale, j // scale][1] * 1.5
            self.image[i, j][2] = ti.max(
                self.concentrations[i // scale, j // scale][1] - 0.3, 0.0
            )

    def gui_loop(self, scale: int = 1, speed: int = 1):
        scaled_size = self.size * scale
        gui = ti.GUI("Gray-Scott Reaction-Diffusion", res=scaled_size, fast_gui=True)
        self.image = ti.Vector.field(3, float, shape=(scaled_size, scaled_size))

        while gui.running:
            for _ in range(speed):
                self.step_diffusion()
            self.draw(scale)
            gui.set_image(self.image)
            gui.show()

    def save_frames(
        self, frames: int | list[int], name: str, scale: int = 1, local: bool = True
    ):
        """
        Saves the given frames from this simulation to png's.
        """
        if isinstance(frames, int):
            frames = [frames]

        gui = ti.GUI("Saving private String", res=self.size, show_gui=False)

        for i in range(max(frames) + 1):
            self.step_diffusion()
            if i in frames:
                frames.remove(i)
                self.draw(scale)
                gui.set_image(self.image)
                gui.show(
                    f"./local/{name}_{i:06d}.png"
                    if local
                    else f"./figures/{name}_{i:06d}.png"
                )


if __name__ == "__main__":
    N = 400
    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 4 - 5 : N // 4 + 5, N // 4 - 5 : N // 4 + 5] = 0.25
    # v_concentration += np.random.rand(*v_concentration.shape) * 0.1

    gray_scott = GrayScott(N, DT=1.0, DX=1.0, DU=0.16, DV=0.08, F=0.035, K=0.060)
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=2, speed=SPEED)

    gray_scott = GrayScott(N, DT=1.0, DX=1.0, DU=0.16, DV=0.08, F=0.030, K=0.060)
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=2, speed=SPEED)

    gray_scott = GrayScott(N, DT=1.0, DX=1.0, DU=0.16, DV=0.08, F=0.039, K=0.060)
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )
    gray_scott.gui_loop(scale=2, speed=SPEED)
