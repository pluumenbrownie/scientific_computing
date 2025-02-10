import taichi as ti 
import taichi.math as tm
from string_functions import init_string_1, init_string_2, init_string_3, update_string


ti.init()


# init boilerplate
N: int = 1_000
dx: float = 1/N
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
        

init_string_3()
update_string()
        
gui = ti.GUI("Vibrating String", res=resolution) #type:ignore

i = 0
while gui.running:  
    update_gui()
    gui.set_image(pixels)
    gui.show()
    i += 1