import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes_lewiner
import plotly.figure_factory as FF
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
from radiomics import featureextractor
import SimpleITK as sitk
import scipy.spatial.distance as distance

def get_subdirectories(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def calculate_nodule_volume(nodule_arr, image):
    non_zero_voxels = np.count_nonzero(nodule_arr)
    spacing = image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    nodule_volume = non_zero_voxels * voxel_volume
    return nodule_volume

def calculate_fractal_dimension(nodule_arr):
    # Only for 2d image
    assert(len(nodule_arr.shape) == 3)

    # Transform nodule_arr into a binary array
    nodule_arr = (nodule_arr > 0)

    # Minimal and maximal box sizes
    sizes = 2**np.arange(3, 10)

    # Box counting
    counts = []
    for size in sizes:
        count = 0
        for x in range(0, nodule_arr.shape[0] - size + 1, size):
            for y in range(0, nodule_arr.shape[1] - size + 1, size):
                for z in range(0, nodule_arr.shape[2] - size + 1, size):
                    if np.any(nodule_arr[x:x+size, y:y+size, z:z+size]):
                        count += 1
        counts.append(count)

    # Add small constant to avoid zero values in log
    counts = np.array(counts) + 1e-10

    # Fit the sizes and counts to a linear equation in log-log scale
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]

def compute_nodule_area(nodule_mask, voxel_spacing):
    area = np.sum(nodule_mask) * voxel_spacing[0] * voxel_spacing[1]
    return area

def compute_features(image, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")

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

def get_calcification_spiculation_features(image_arr, mask_arr, nodule_diameter):
    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)

    features = compute_features(image, mask)

    correlation, entropy, contrast, energy, homegenetiy = get_glcm_features(image_arr, mask_arr)
    calcification = correlation
    spiculation = features['original_glrlm_ShortRunEmphasis']

    if nodule_diameter <= 10:
        D1 = -0.665 * correlation + 3.194 * entropy - 2.359 * contrast + 3.194 * energy - 1.986 * homegenetiy
        if D1 > 0:
            type_of_nodule = "Malign"
        else:
            type_of_nodule = "Benign"
    if nodule_diameter <= 20:
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

    return calcification, spiculation, type_of_nodule

def return_fig(images, threshold, step_size):
    p = images.transpose(2, 1, 0)
    verts, faces, _, _ = marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    x, y, z = zip(*verts)
    colormap = ['rgb(255, 192, 203)', 'rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap=colormap, simplices=faces, backgroundcolor='rgb(125, 125, 125)', title="3D Visualization of the CT Scan")
    return fig


data_folder = "C:\\Users\\fabi2\\OneDrive\\Documents\\GitHub\\4D_CT_Scan\\Visual\\data\\converted_nrrds\\"
subdirectories = get_subdirectories(data_folder)
png_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\data\\images_quick_check\\"

def get_png_files(folder):
    png_folder = os.path.join(folder + "_GTV-1_mask\\GTV-1_mask")
    png_files = [os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith(".png")]
    png_files.sort()
    return png_files

# Initialize imgs with the first subdirectory images
if subdirectories:
    initial_selected_folder = subdirectories[0]
    initial_file_struct = None
    for root, _, files in os.walk(os.path.join("C:\\Users\\fabi2\\OneDrive\\Desktop\\data\\images_quick_check\\", initial_selected_folder)):
        initial_file_struct = (root, files)
        break
    if initial_file_struct:
        initial_root, initial_images = initial_file_struct
        imgs = [np.array(Image.open(os.path.join(initial_root, img))) for img in initial_images]

app = dash.Dash(__name__, prevent_initial_callbacks='initial_duplicate')

initial_fig = None
if subdirectories:
    initial_image_nrrd_file = os.path.join(data_folder, subdirectories[0], "image.nrrd")
    initial_mask_nrrd_file = os.path.join(data_folder, subdirectories[0], "GTV-1_mask.nrrd")

    initial_image = sitk.ReadImage(initial_image_nrrd_file)
    initial_mask = sitk.ReadImage(initial_mask_nrrd_file)

    initial_image_arr = sitk.GetArrayFromImage(initial_image)
    initial_mask_arr = sitk.GetArrayFromImage(initial_mask)

    initial_image_normalized = (initial_image_arr - np.min(initial_image_arr)) / (np.max(initial_image_arr) - np.min(initial_image_arr))
    initial_nodule_arr = initial_image_normalized * initial_mask_arr

    initial_fig = return_fig(initial_nodule_arr, threshold=0.25, step_size=1)

app.layout = html.Div(
    children=[
        html.H1(children="CT scan 3D Visualization", style={"font-size": "32px"}),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Graph(id='graph-with-selector', figure=initial_fig),
                    ],
                    style={"display": "inline-block", "vertical-align": "top", "width": "40%"}  # Update the style here
                ),
                html.Div(
                    children=[
                        html.P(id='nodule-volume-info', style={"font-size": "32px"}),
                        html.P(id='fractal-dimension-info', style={"font-size": "32px"}),
                        html.P(id='nodule-area-info', style={"font-size": "32px"}),
                        html.P(id='calcification-info', style={"font-size": "32px"}),
                        html.P(id='spiculation-info', style={"font-size": "32px"}),
                        html.P(id='type-of-nodule-info', style={"font-size": "32px"}),
                    ],
                    style={"display": "inline-block", "vertical-align": "top", "margin-left": "2px"}  # Update the style here
                ),
            ],
            style={"display": "block"}  # Update the style here
        ),
        dcc.Dropdown(
            id='folder-selector',
            options=[{'label': subdir, 'value': i} for i, subdir in enumerate(subdirectories)] + [{'label': 'None', 'value': -1}],
            value=0,  # Set the initial value to -1, representing no selection
            style={"max-width": "600px", "margin": "10px"},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Slider(id='png-slider', min=0, max=0, value=0, step=1),
                        html.Img(id='png-viewer', src=''),
                    ],
                    style={"display": "inline-block", "vertical-align": "top", "width": "30%"}  # Update the style here
                ),
            ],
            style={"display": "block", "margin-top": "20px"}  # Update the style here
        ),
    ]
)

@app.callback(
    [Output('graph-with-selector', 'figure'),
     Output('nodule-volume-info', 'children'),
     Output('fractal-dimension-info', 'children'),
     Output('nodule-area-info', 'children'),
     Output('calcification-info', 'children'),
     Output('spiculation-info', 'children'),
     Output('type-of-nodule-info', 'children'),
     Output('png-slider', 'max'),
     Output('png-viewer', 'src', allow_duplicate=True)],
    [Input('folder-selector', 'value'),
     Input('png-slider', 'value')],
)

def update_figure(selected_folder_index, slider_value):
    if selected_folder_index != -1:
        selected_folder = subdirectories[selected_folder_index]
        image_nrrd_file = os.path.join(data_folder, selected_folder, "image.nrrd")
        mask_nrrd_file = os.path.join(data_folder, selected_folder, "GTV-1_mask.nrrd")

        image = sitk.ReadImage(image_nrrd_file)
        mask = sitk.ReadImage(mask_nrrd_file)

        image_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask)

        image_normalized = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        nodule_arr = image_normalized * mask_arr

        nodule_volume = calculate_nodule_volume(nodule_arr, image)
        nodule_fractal_dimension = calculate_fractal_dimension(nodule_arr)

        updated_fig = return_fig(nodule_arr, threshold=0.25, step_size=1)

        volume_info = f"Nodule volume: {nodule_volume:.2f} mm³"
        fractal_info = f"Fractal dimension: {nodule_fractal_dimension:.2f}"

        voxel_spacing = image.GetSpacing()
        nodule_area = compute_nodule_area(mask_arr, voxel_spacing)
        area_info = f"Nodule area: {nodule_area:.2f} mm²"

        nodule_diameter = calculate_max_distance(mask_arr, voxel_spacing)
        calcification, spiculation, type_of_nodule = get_calcification_spiculation_features(image_arr, mask_arr, nodule_diameter)
        calcification_info = f"Calcification: {calcification:.4f}"
        spiculation_info = f"Spiculation: {spiculation:.4f}"
        type_of_nodule_info = f"Nodule type: {type_of_nodule}"

        nodule_diameter = calculate_max_distance(mask_arr, voxel_spacing)

        png_files = get_png_files(os.path.join(png_folder, selected_folder))
        if png_files:
            max_slider_value = len(png_files) - 1
            png_file_path = png_files[slider_value]
            with open(png_file_path, "rb") as f:
                image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes)
            png_src = f"data:image/png;base64,{encoded_image.decode()}"
        else:
            max_slider_value = 0
            png_src = ''

    else:
        updated_fig = return_fig(np.zeros((2, 2, 2)), threshold=0.25, step_size=1)
        volume_info = f"Nodule volume: 0 mm³"
        fractal_info = f"Fractal dimension: 0"
        area_info = f"Nodule area: 0 mm²"
        calcification_info = f"Calcification: 0"
        spiculation_info = f"Spiculation: 0"
        type_of_nodule_info = f"Nodule type: None"
        max_slider_value = -1  # Initialize to -1 if no subdirectory is selected
        png_src = ''

    return updated_fig, volume_info, fractal_info, area_info, calcification_info, spiculation_info, type_of_nodule_info, max_slider_value, png_src

@app.callback(
    Output('png-viewer', 'src'),
    Input('png-slider', 'value'),
    State('folder-selector', 'value'),
    prevent_initial_call='initial_duplicate'  # Add this parameter
)
def update_png_viewer(slider_value, selected_folder_index):
    if selected_folder_index != -1:
        selected_folder = subdirectories[selected_folder_index]
        png_files = get_png_files(os.path.join(png_folder, selected_folder))
        if png_files:
            png_file_path = png_files[slider_value]
            with open(png_file_path, "rb") as f:
                image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes)
            return f"data:image/png;base64,{encoded_image.decode()}"
    return ''


if __name__ == "__main__":
    app.run_server(debug=True, port=80)
