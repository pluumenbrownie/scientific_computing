import taichi as ti 
import taichi.math as tm
import numpy as np
from string_functions import init_string_1, init_string_2, init_string_3, update_string


ti.init()


# init boilerplate
N: int = 1_000
dx: float = 1/N
dt: float = 0.001
amplitude = ti.field(float, shape = N)
amplitude_previous = ti.field(float, shape = N)

# for gui
resolution = (512, 256)
pixels = ti.field(float, shape=resolution)


@ti.kernel
def update_gui():
    float_N = float(N)
    for i, j in pixels:
        pixels[i, j] = 1
    for i in amplitude:
        i_pixel = tm.round(i / float_N * resolution[0], dtype=int)
        j_pixel = tm.round(amplitude[i] * 128 + 128, dtype=int)
        pixels[i_pixel, j_pixel] = 0


def save_video(frames: int, location: str):
    init_string_3()
    video_manager = ti.tools.VideoManager(output_dir=location, framerate=24, automatic_build=False, video_filename="test_string")
    for _ in range(frames):
        update_string()
        update_gui()
        img = pixels.to_numpy()
        video_manager.write_frame(img)

    video_manager.make_video(mp4=True, gif=False)
    video_manager.clean_frames()


def get_wave_points():
    width, height = resolution
    x_coords = np.linspace(0, 1, N) 
    y_coords = 0.5 + amplitude.to_numpy() * 0.2 
    return np.stack([x_coords, y_coords], axis=1)
        

gui = ti.GUI("Vibrating String", res=resolution) #type:ignore
init_string_3()

i = 0
while gui.running:
    update_string()
    wave_points = get_wave_points()
    gui.clear(0xFFFFFF)
    gui.lines(wave_points[:-1], wave_points[1:], radius=1, color=0xFF0000) 
    update_gui()
    gui.show()
    i += 1
