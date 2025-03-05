import numpy as np
from gray_scott import GrayScott


def gray_scott_experiments():
    N = 200
    parameters = [(0.035, 0.060)]

    u_concentration = np.full((N, N), 0.5, dtype=np.float32)
    v_concentration = np.zeros_like(u_concentration)
    v_concentration[N // 4 - 5 : N // 4 + 5, N // 4 - 5 : N // 4 + 5] = 0.25
    v_concentration += np.random.rand(*v_concentration.shape) * 0.1

    for F, K in parameters:
        gray_scott = GrayScott(N, F=F, K=K)
        gray_scott.init_concentration(
            u_concentration=u_concentration, v_concentration=v_concentration
        )
        gray_scott.save_frames([1000], name=f"f{F}K{K}")
