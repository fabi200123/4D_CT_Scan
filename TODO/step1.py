import os
import glob
import SimpleITK as sitk
import numpy as np
import scipy.ndimage


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpy_image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpy_image, origin, spacing


def resample_image(image, spacing, new_spacing=[1, 1, 1]):
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    resampled_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return resampled_image, new_spacing


def normalize_hu(image, min_hu=-1000, max_hu=400):
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    return image


def segment_lung_mask(image, threshold=-400):
    binary_image = np.array(image > threshold, dtype=np.int8) + 1
    labels, num_labels = scipy.ndimage.measurements.label(binary_image)

    # Pick the label with the largest region (excluding label 0, which is the background)
    largest_label = np.argmax(np.bincount(labels.ravel())[1:]) + 1

    return np.where(labels == largest_label, 1, 0)

def preprocess_luna16_data(input_folder, output_folder):
    subset_folders = [os.path.join(input_folder, f'subset{i}') for i in range(10)]

    for subset_folder in subset_folders:
        file_list = glob.glob(os.path.join(subset_folder, "*.mhd"))

        for f in file_list:
            print("Processing file:", f)
            image, origin, spacing = load_itk_image(f)
            resampled_image, new_spacing = resample_image(image, spacing)
            normalized_image = normalize_hu(resampled_image)
            lung_mask = segment_lung_mask(normalized_image)

            output_filename = os.path.basename(f).replace(".mhd", ".npy")
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, lung_mask * normalized_image)


input_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\"
output_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\Output"
preprocess_luna16_data(input_folder, output_folder)
