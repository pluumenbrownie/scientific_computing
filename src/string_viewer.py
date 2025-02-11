import taichi as ti
import taichi.math as tm
import numpy as np
import os
from string_functions import init_string_1, init_string_2, init_string_3, update_string


ti.init()


# init boilerplate
N: int = 1_000
dx: float = 1 / N
dt: float = 0.001
amplitude_next = ti.field(float, shape=N)
amplitude = ti.field(float, shape=N)
amplitude_previous = ti.field(float, shape=N)

# for gui
resolution = (512, 256)


def update_gui(gui: ti.GUI):
    wave_points = get_wave_points()
    gui.clear(0xFFFFFF)
    gui.lines(wave_points[:-1], wave_points[1:], radius=1, color=0xFF0000)


def save_video(frames: int, location: str = "./local"):
    init_string_3()
    video_manager = ti.tools.VideoManager(
        output_dir=location,
        framerate=24,
        automatic_build=False,
        video_filename="test_string",
    )
    gui = ti.GUI("Saving private String", res=resolution, show_gui=False)  # type:ignore
    for _ in range(frames):
        update_string()
        update_gui(gui)
        video_manager.write_frame(gui.get_image())

    video_manager.make_video(mp4=True, gif=False)
    # video_manager.clean_frames() is known to be broken https://github.com/taichi-dev/taichi/issues/6936
    frame_folder = location + "/frames"
    for fn in os.listdir(frame_folder):
        if fn.endswith('.png'):
            os.remove(frame_folder + '/' + fn) 
    os.rmdir(frame_folder)


def get_wave_points():
    width, height = resolution
    x_coords = np.linspace(0, 1, N)
    y_coords = 0.5 + amplitude.to_numpy() * 0.2
    return np.stack([x_coords, y_coords], axis=1)


def run_gui():
    gui = ti.GUI("Vibrating String", res=resolution)  # type:ignore
    init_string_3()

    while gui.running:
        update_string()
        update_gui(gui)
        gui.show()


if __name__ == "__main__":
    # run_gui()
    save_video(100)
