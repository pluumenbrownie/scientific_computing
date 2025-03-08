import numpy as np
from gray_scott import GrayScott
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def create_model(N: int, F: float, K: float, noise: bool):
    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 4 - 5 : N // 4 + 5, N // 4 - 5 : N // 4 + 5] = 0.25
    if noise:
        v_concentration += np.random.rand(*v_concentration.shape) * 0.01
    gray_scott = GrayScott(N, F=F, K=K)
    gray_scott.init_concentration(
        u_concentration=u_concentration, v_concentration=v_concentration
    )

    return gray_scott


def single_frames():
    """
    Let taichi export the given steps for the given models.
    """
    N = 200
    # the parameters for the models we want to save
    parameters = [
        (0.035, 0.060),
        (0.030, 0.060),
        (0.039, 0.060),
        (0.029, 0.060),
    ]
    # the steps we want to save
    saved_steps = [2500, 5000, 7500]

    for F, K in parameters:
        gray_scott = create_model(N, F, K, False)
        gray_scott.save_frames(saved_steps, name=f"f{F}K{K}")


def pyplot_combined():
    N = 150
    # the parameters for the models we want to save
    parameters = [
        (0.028, 0.060, False),
        (0.028, 0.060, True),
        (0.030, 0.060, False),
        (0.030, 0.060, True),
        (0.035, 0.060, False),
        (0.035, 0.060, True),
        (0.039, 0.060, False),
        (0.039, 0.060, True),
    ]
    # the steps we want to save
    saved_steps = [2500, 5000, 7500]
    saved_steps = sorted(saved_steps)

    fig = plt.figure(figsize=(4, 10), layout="compressed")
    axes = fig.subplots(
        nrows=len(parameters),
        ncols=len(saved_steps),
        subplot_kw={"xticks": [], "yticks": []},
    )
    storage = np.zeros((len(parameters), len(saved_steps), N, N), dtype=np.float32)

    for row, (F, K, noise) in enumerate(parameters):
        gray_scott = create_model(N, F, K, noise)
        for col, step in enumerate(saved_steps):
            gray_scott.run_until_step(step)
            storage[row, col] = gray_scott.concentrations.to_numpy()[:, :, 1]

    norm = Normalize(vmin=storage.min(), vmax=storage.max())
    images = []
    for row, (F, K, noise) in enumerate(parameters):
        for col, step in enumerate(saved_steps):
            images.append(axes[row, col].imshow(storage[row, col], norm=norm))
            if col == 1 and noise:
                axes[row, col].set_xlabel(f"$f={F}, k={K}$")
            if col == 0:
                axes[row, col].set_ylabel(f"{'' if noise else 'no '}noise")

    # fig.suptitle("Gray-Scott reactions for varying $f$ and $k$")
    fig.colorbar(images[0], ax=axes, orientation="horizontal", pad=0.003)
    plt.savefig("local/combined.png")


if __name__ == "__main__":
    pyplot_combined()
