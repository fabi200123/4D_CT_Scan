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
from .calculate_nodule_dimensions import calculate_nodule_volume, calculate_fractal_dimension, compute_nodule_area, calculate_max_distance

def compute_features(image, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glcm")

    features = extractor.execute(image, mask)

    return features

def get_glcm_features(image_arr, mask_arr):
    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)

    features = compute_features(image, mask)

    correlation = features["original_glcm_Correlation"]
    entropy = features["original_glcm_JointEntropy"]
    contrast = features["original_glcm_Contrast"]
    energy = features["original_glcm_JointEnergy"]
    homegenetiy = features["original_glcm_Id"]

    return correlation, entropy, contrast, energy, homegenetiy

def get_spiculation(image_arr, mask_arr):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glrlm")

    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)

    features = extractor.execute(image, mask)

    spiculation = features['original_glrlm_ShortRunEmphasis']
    
    return spiculation

def get_calcification_nodule_type_features(image_arr, mask_arr, nodule_diameter):
    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)

    features = compute_features(image, mask)

    correlation, entropy, contrast, energy, homegenetiy = get_glcm_features(image_arr, mask_arr)
    calcification = correlation

    if nodule_diameter <= 10:
        D1 = -0.665 * correlation + 3.194 * entropy - 2.359 * contrast + 3.194 * energy - 1.986 * homegenetiy
        if D1 > 0:
            type_of_nodule = "Malign"
        else:
            type_of_nodule = "Benign"
    elif nodule_diameter <= 20:
        D1 = 0.137 * correlation - 0.562 * entropy + 2.454 * contrast - 1.776* energy + 2.938* homegenetiy
        if D1 < 0:
            type_of_nodule = "Malign"
        else:
            type_of_nodule = "Benign"
    else:
        D1 = -0.465 * correlation + 0.133 * entropy + 2.231 * contrast - 1.344 * energy + 3.288 * homegenetiy
        if D1 < 0:
            type_of_nodule = "Malign"
        else:
            type_of_nodule = "Benign"
    return calcification, type_of_nodule

def get_all_features(data_folder, subdirectories):
    nodule_volume = []
    nodule_fractal_dimension = []
    nodule_area = []
    calcification = []
    spiculation = []
    type_of_nodule = []
    for selected_folder_index, selected_folder in enumerate(subdirectories):
        image_nrrd_file = os.path.join(data_folder, selected_folder, "image.nrrd")
        mask_nrrd_file = os.path.join(data_folder, selected_folder, "GTV-1_mask.nrrd")

        image = sitk.ReadImage(image_nrrd_file)
        mask = sitk.ReadImage(mask_nrrd_file)

        image_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask)

        image_normalized = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        nodule_arr = image_normalized * mask_arr

        nodule_volume.append(calculate_nodule_volume(nodule_arr, image))
        nodule_fractal_dimension.append(calculate_fractal_dimension(nodule_arr))

        voxel_spacing = image.GetSpacing()
        nodule_area.append(compute_nodule_area(mask_arr, voxel_spacing))

        nodule_diameter = calculate_max_distance(mask_arr, voxel_spacing)
        calcification_value, type_of_nodule_value = get_calcification_nodule_type_features(image_arr, mask_arr, nodule_diameter)
        calcification.append(calcification_value)
        type_of_nodule.append(type_of_nodule_value)
        spiculation.append(get_spiculation(image_arr, mask_arr))
    
    return nodule_volume, nodule_fractal_dimension, nodule_area, calcification, spiculation, type_of_nodule
