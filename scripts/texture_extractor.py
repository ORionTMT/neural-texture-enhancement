from PIL import Image
import os
import numpy as np

map_name = 'd2_prison_04_d'
material_color_hex = 'baec46'

material_color = tuple(int(material_color_hex[i:i+2], 16) for i in (0, 2, 4)) # Convert to tuple of ints
tolerance = 4
texture_size = 128

texture = (np.empty((texture_size, texture_size, 3), dtype=float) * 255).astype(np.uint8)
capture_depth = np.zeros((texture_size, texture_size), dtype=int)

def is_same_color(color1, color2, tolerance):
    return all(abs(color1[i] - color2[i]) <= tolerance for i in range(3))

for img_name in os.listdir(f'./{map_name}/color'):
    color = Image.open(f'./{map_name}/color/{img_name}').convert('RGB')
    materials = Image.open(f'./{map_name}/materials/{img_name}')
    uv = Image.open(f'./{map_name}/uv/{img_name}')
    depth = Image.open(f'./{map_name}/depth/{img_name}').convert('L')

    for x in range(color.width):
        for y in range(color.height):
            if is_same_color(materials.getpixel((x, y)), material_color, tolerance):
                uv_sample = uv.getpixel((x, y))
                u = uv_sample[0] / 256 * texture_size
                v = uv_sample[1] / 256 * texture_size

                if capture_depth[int(v), int(u)] == 0.0 or depth.getpixel((x, y)) < capture_depth[int(v), int(u)]:
                    texture[int(v), int(u)] = color.getpixel((x, y))
                    capture_depth[int(v), int(u)] = depth.getpixel((x, y))

    Image.fromarray(texture).save(f'texture.png')
    print(f'Processed {img_name}')
