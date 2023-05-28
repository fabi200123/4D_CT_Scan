import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
from radiomics import featureextractor
import SimpleITK as sitk
import scipy.spatial.distance as distance
import numpy as np
from .FractalDimension import fractal_dimension
import matplotlib.pyplot as plt

def calculate_nodule_volume(nodule_arr, image):
    non_zero_voxels = np.count_nonzero(nodule_arr)
    spacing = image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    nodule_volume = non_zero_voxels * voxel_volume
    return nodule_volume

# def calculate_fractal_dimension(nodule_arr):
#     # Only for 2d image
#     assert(len(nodule_arr.shape) == 3)

#     # Transform nodule_arr into a binary array
#     nodule_arr = (nodule_arr > 0)

#     # Minimal and maximal box sizes
#     sizes = 2**np.arange(3, 10)

#     # Box counting
#     counts = []
#     for size in sizes:
#         count = 0
#         for x in range(0, nodule_arr.shape[0] - size + 1, size):
#             for y in range(0, nodule_arr.shape[1] - size + 1, size):
#                 for z in range(0, nodule_arr.shape[2] - size + 1, size):
#                     if np.any(nodule_arr[x:x+size, y:y+size, z:z+size]):
#                         count += 1
#         counts.append(count)

#     # Add small constant to avoid zero values in log
#     counts = np.array(counts) + 1e-10

#     # Fit the sizes and counts to a linear equation in log-log scale
#     coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

#     return -coeffs[0]

def calculate_fractal_dimension(nodule_arr):
    # Only for 2d image
    assert(len(nodule_arr.shape) == 3)

    # Transform nodule_arr into a binary array
    nodule_arr = (nodule_arr > 0)

    # Compute the fractal dimension using the module's function
    fractalDimension = fractal_dimension(nodule_arr)

    return fractalDimension

def compute_nodule_area(nodule_mask, voxel_spacing):
    area = np.sum(nodule_mask) * voxel_spacing[0] * voxel_spacing[1]
    return area

def calculate_max_distance(mask, voxel_spacing):
    max_distance_axial = 0
    max_distance_sagittal = 0
    max_distance_coronal = 0

    for z in range(mask.shape[0]):
        axial_slice = mask[z, :, :]
        coords = np.argwhere(axial_slice > 0)
        if coords.size > 0:
            coords_mm = coords * voxel_spacing[1:]
            max_distance_axial = max(max_distance_axial, np.max(distance.pdist(coords_mm)))

    for y in range(mask.shape[1]):
        sagittal_slice = mask[:, y, :]
        coords = np.argwhere(sagittal_slice > 0)
        if coords.size > 0:
            coords_mm = coords * [voxel_spacing[0], voxel_spacing[2]]
            max_distance_sagittal = max(max_distance_sagittal, np.max(distance.pdist(coords_mm)))

    for x in range(mask.shape[2]):
        coronal_slice = mask[:, :, x]
        coords = np.argwhere(coronal_slice > 0)
        if coords.size > 0:
            coords_mm = coords * voxel_spacing[:2]
            max_distance_coronal = max(max_distance_coronal, np.max(distance.pdist(coords_mm)))

    return max(max_distance_axial, max_distance_sagittal, max_distance_coronal)