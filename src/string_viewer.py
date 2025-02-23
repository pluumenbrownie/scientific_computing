from typing import Callable
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
from string_functions import init_string_1, init_string_2, init_string_3, update_string, run_simulation


ti.init()


# init boilerplate
N: int = 1000
dx: float = 1 / N
dt: float = 0.001
amplitude_next = ti.field(float, shape=N)
amplitude = ti.field(float, shape=N)
amplitude_previous = ti.field(float, shape=N)

# for gui
resolution = (512, 256)


def update_gui(gui: ti.GUI):
    """
    Clears the current gui viewport and draws the string as a series of lines.
    """
    wave_points = get_wave_points()
    gui.clear(0xFFFFFF)
    gui.lines(wave_points[:-1], wave_points[1:], radius=1, color=0xFF0000)


def save_video(
    frames: int,
    init_fn: Callable,
    name: str,
    location: str = "./local",
    step_size: int = 1,
    fps: int = 60,
):
    """
    Save a simulation of a string to an mp4 video.
    ```
    save_video(400, init_string_1, step_size=4)
    ```
    """
    init_fn()
    video_manager = ti.tools.VideoManager(
        output_dir=location,
        framerate=fps,
        automatic_build=True,
        video_filename=name,
    )
    gui = ti.GUI("Saving private String", res=resolution, show_gui=False)  # type:ignore
    for _ in range(frames):
        for _ in range(step_size):
            update_string()
            update_gui(gui)
        video_manager.write_frame(gui.get_image())

    video_manager.make_video(mp4=True, gif=False)
    # video_manager.clean_frames() is known to be broken https://github.com/taichi-dev/taichi/issues/6936
    frame_folder = os.path.join(location, "frames")
    for fn in os.listdir(frame_folder):
        if fn.endswith(".png"):
            os.remove(frame_folder + "/" + fn)
    os.rmdir(frame_folder)


def save_frames(frames: int | list[int], init_fn: Callable, name: str, location: str = "./local"):
    """
    Saves the given frames from a given simulation to png's. 
    """
    if isinstance(frames, int):
        frames = [frames]

    init_fn()
    gui = ti.GUI("Saving private String", res=resolution, show_gui=False)  # type:ignore

    for i in range(max(frames)+1):
        update_string()
        update_gui(gui)
        if i in frames:
            frames.remove(i)
            gui.show(f"{name}_{i:06d}.png")


def get_wave_points():
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

def plot_figure(result):
    x = np.linspace(0, 1, N)
    time = [100,200,300,400,500,600]
    """fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

    for ax, t_idx in zip(axes, time):
        for (key, data) in result.items():
            ax.plot(x, data[t_idx], label=f'{key}')
        ax.set_title(f'Time step {t_idx}')
        ax.set_xlabel('x')
        ax.set_ylabel('Amplitude')
        ax.set_ylim([-1.2, 1.2])
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig("local/1.1b.png", dpi=300)
    plt.show()"""

    for key, data in result.items(): 
        for t in time: 
            plt.plot(x, data[t], label=f'Time step {t}')
        plt.title(f'Initial Condition: {key}')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.savefig(f"local/{key}_1.1b.png", dpi=300)
        plt.show()



if __name__ == "__main__":
    # run_gui()
    # save_frames([0, 100, 200], init_string_3, "string_3")
    # save_video(50, init_string_3, name = "string3", step_size=4)
    results = {
    'initial condition 1': run_simulation(init_string_1),
    'initial condition 2': run_simulation(init_string_2),
    'initial condition 3': run_simulation(init_string_3),}
    plot_figure(results)
    
    save_video(1000, init_string_1, "string_1")
    save_video(1000, init_string_2, "string_2")
    save_video(1000, init_string_3, "string_3")

    
