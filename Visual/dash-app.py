import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes_lewiner
import plotly.figure_factory as FF
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

def get_subdirectories(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def calculate_nodule_volume(nodule_arr, image):
    non_zero_voxels = np.count_nonzero(nodule_arr)
    spacing = image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    nodule_volume = non_zero_voxels * voxel_volume
    return nodule_volume


def return_fig(images, threshold, step_size):
    p = images.transpose(2, 1, 0)
    verts, faces, _, _ = marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    x, y, z = zip(*verts)
    colormap = ['rgb(255, 192, 203)', 'rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap=colormap, simplices=faces, backgroundcolor='rgb(125, 125, 125)', title="3D Visualization of the CT Scan")
    return fig


data_folder = "C:\\Users\\fabi2\\OneDrive\\Documents\\GitHub\\4D_CT_Scan\\Visual\\data\\converted_nrrds\\"
subdirectories = get_subdirectories(data_folder)

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

app = dash.Dash(__name__)

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
        html.H1(children="CT Scan"),
        html.P(children="CT scan 3D Visualization"),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Graph(id='graph-with-selector', figure=initial_fig),
                    ],
                    style={"display": "inline-block", "vertical-align": "top", "width": "70%"}  # Update the style here
                ),
                html.Div(
                    children=[
                        html.P(id='nodule-volume-info'),
                    ],
                    style={"display": "inline-block", "vertical-align": "left", "margin-left": "2px"}  # Update the style here
                ),
            ],
            style={"display": "block"}  # Update the style here
        ),
        dcc.Dropdown(
            id='folder-selector',
            options=[{'label': subdir, 'value': i} for i, subdir in enumerate(subdirectories)] + [{'label': 'None', 'value': -1}],
            value=-1,  # Set the initial value to -1, representing no selection
            style={"max-width": "600px", "margin": "10px"},
        ),
    ]
)

@app.callback(
    [Output('graph-with-selector', 'figure'),
     Output('nodule-volume-info', 'children'),],
    Input('folder-selector', 'value'),
)
def update_figure(selected_folder_index):
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

        updated_fig = return_fig(nodule_arr, threshold=0.25, step_size=1)
        volume_info = f"Nodule volume: {nodule_volume:.2f} mm³"
    else:
        updated_fig = return_fig(np.zeros((2, 2, 2)), threshold=0.25, step_size=1)
        volume_info = f"Nodule volume: 0 mm³"

    return updated_fig, volume_info

if __name__ == "__main__":
    app.run_server(debug=True, port=80)
