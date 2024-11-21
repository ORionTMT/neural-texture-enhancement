from PIL import Image
import numpy as np
import os

color_path = 'd2_coast_04_d/color/'
materials_path = 'd2_coast_04_d/materials/'
output_path = 'd2_coast_04_d/lofi/'

def load_image_np(path: str) -> np.ndarray:
    return np.array(Image.open(path))

def get_main_colors(image: np.ndarray) -> np.ndarray:
    color_dict = {} # For each color, we will count how many pixels have that color

    for pixel in image.reshape(-1, image.shape[-1]):
        color = tuple(pixel)
        if color in color_dict:
            color_dict[color] += 1
        else:
            color_dict[color] = 1

    # Sort the colors by frequency
    sorted_colors = sorted(color_dict.items(), key=lambda x: x[1], reverse=True)

    # Discard colors that make up a tiny fraction of the image
    total_pixels = image.shape[0] * image.shape[1]
    sorted_colors = [(color, count) for color, count in sorted_colors if count > total_pixels * 0.00001]

    # Return only the colors
    return np.array([color for color, _ in sorted_colors])

def get_mask(image: np.ndarray, color: np.ndarray, tolerance: int) -> np.ndarray:
    image = np.array(image)
    color = np.array(color)

    diff = np.abs(image - color)

    mask = np.all(diff <= tolerance, axis=-1)

    return mask

def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    return image[::factor, ::factor]

def get_average_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Check if the mask is empty
    if not mask.any():
        return np.zeros(image.shape[-1])

    return image[mask].mean(axis=0)

for image in os.listdir(color_path):
    print("Processing", image)
    color = load_image_np(color_path + image)
    materials = load_image_np(materials_path + image)

    mat_colors = get_main_colors(downsample(materials, 8))
    print("Found", len(mat_colors), "materials")

    print("Building masks")
    masks = [get_mask(materials, mat_color, 5) for mat_color in mat_colors]

    print("Assigning colors")
    average_colors = [get_average_color(color, mask) for mask in masks]

    print("Building lofi image")
    lofi = np.zeros_like(color)

    for mask, color in zip(masks, average_colors):
        lofi[mask] = color

    Image.fromarray(lofi).save(output_path + image)

