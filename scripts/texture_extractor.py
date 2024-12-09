'''
Utility script to extract a texture from an image sequence
'''

from PIL import Image
import os
import numpy as np

map_name = 'd1_trainstation_02_d'
material_color = (0x23, 0x8d, 0xd6)
tolerance = 4
texture_size = 64

texture = (np.empty((texture_size, texture_size, 3), dtype=float) * 255).astype(np.uint8)

def is_same_color(color1, color2, tolerance):
    return all(abs(color1[i] - color2[i]) <= tolerance for i in range(3))

for img_name in os.listdir(f'./{map_name}/color'):
    color = Image.open(f'./{map_name}/color/{img_name}')
    materials = Image.open(f'./{map_name}/materials/{img_name}')
    uv = Image.open(f'./{map_name}/uv/{img_name}')

    for x in range(color.width):
        for y in range(color.height):
            if is_same_color(materials.getpixel((x, y)), material_color, tolerance):
                uv_sample = uv.getpixel((x, y))
                u = uv_sample[0] / 256 * texture_size
                v = uv_sample[1] / 256 * texture_size

                texture[int(v), int(u)] = color.getpixel((x, y))

    Image.fromarray(texture).save(f'texture.png')
    print(f'Processed {img_name}')
