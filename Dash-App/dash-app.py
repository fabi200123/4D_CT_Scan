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
import plotly.graph_objects as go

from modules.nodule_features import get_all_features

def get_subdirectories(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def return_fig(images, threshold, step_size):
    p = images.transpose(2, 1, 0)
    verts, faces, _, _ = marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    x, y, z = zip(*verts)
    colormap = ['rgb(255, 192, 203)', 'rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap=colormap, simplices=faces, backgroundcolor='rgb(125, 125, 125)', title="3D Visualization of the CT Scan")
    return fig


data_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\data\\converted_nrrds\\"
subdirectories = get_subdirectories(data_folder)
png_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\data\\images_quick_check\\"

# Initialize the vectors for features
nodule_volume = []
nodule_fractal_dimension = []
nodule_area = []
calcification = []
spiculation = []
type_of_nodule = []

def get_png_files(folder):
    png_folder = os.path.join(folder + "_GTV-1_mask\\GTV-1_mask")
    png_files = [os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith(".png")]
    png_files.sort()
    return png_files

# Initialize imgs with the first subdirectory images
if subdirectories:
    initial_selected_folder = subdirectories[0]
    initial_file_struct = None
    for root, _, files in os.walk(os.path.join("C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\data\\images_quick_check\\", initial_selected_folder)):
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

nodule_volume, nodule_fractal_dimension, nodule_area, calcification, spiculation, type_of_nodule = get_all_features(data_folder, subdirectories) 

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
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[
                                {'label': 'Nodule Volume', 'value': 'nodule-volume'},
                                {'label': 'Fractal Dimension', 'value': 'fractal-dimension'},
                                {'label': 'Nodule Area', 'value': 'nodule-area'},
                                {'label': 'Calcification', 'value': 'calcification'},
                                {'label': 'Spiculation', 'value': 'spiculation'},
                                {'label': 'Nodule type', 'value': 'nodule-type'},
                            ],
                            value='None'  # The default value
                        ),
                        html.Div(id='info-display', children=[
                            html.P(id='info-text', style={"font-size": "32px"}),
                            dcc.Graph(id='info-graph'),
                        ]),
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
     Output('info-text', 'children', allow_duplicate=True),
     # Output('info-graph', 'figure', allow_duplicate=True),
     Output('feature-dropdown', 'value'),
     Output('png-slider', 'max'),
     Output('png-viewer', 'src', allow_duplicate=True)],
    [Input('folder-selector', 'value'),
     Input('png-slider', 'value')],
)

def update_figure(selected_folder_index, slider_value):
    updated_fig = initial_fig
    info_display = f""
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
        updated_fig = return_fig(nodule_arr, threshold=0.25, step_size=1)

        selected_feature = 'nodule-volume'
        info_display = f"Nodule volume: {nodule_volume[selected_folder_index]: .2f} mm³"
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
        info_display = f"None"
        max_slider_value = -1  # Initialize to -1 if no subdirectory is selected
        png_src = ''

    return updated_fig, info_display, selected_feature, max_slider_value, png_src

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

@app.callback(
    [Output('info-text', 'children'),
     Output('info-graph', 'figure')],
    [Input('folder-selector', 'value'),
     Input('feature-dropdown', 'value')],
    prevent_initial_call=True  # Add this parameter
)
def update_info_display(selected_folder_index, selected_feature):
    info_text = ''
    feature_data = []
    selected_feature_data = 0
    if selected_feature == 'nodule-volume':
        info_text = f"Nodule volume: {nodule_volume[selected_folder_index]: .2f} mm³"
        feature_data = nodule_volume
        selected_feature_data = nodule_volume[selected_folder_index]
    elif selected_feature == 'fractal-dimension':
        info_text = f"Fractal dimension: {nodule_fractal_dimension[selected_folder_index]:.2f}"
        feature_data = nodule_fractal_dimension
        selected_feature_data = nodule_fractal_dimension[selected_folder_index]
    elif selected_feature == 'nodule-area':
        info_text = f"Nodule area: {nodule_area[selected_folder_index]:.2f} mm²"
        feature_data = nodule_area
        selected_feature_data = nodule_area[selected_folder_index]
    elif selected_feature == 'calcification':
        info_text = f"Calcification: {calcification[selected_folder_index]:.4f}"
        feature_data = calcification
        selected_feature_data = calcification[selected_folder_index]
    elif selected_feature == 'spiculation':
        info_text = f"Spiculation: {spiculation[selected_folder_index]:.4f}"
        feature_data = spiculation
        selected_feature_data = spiculation[selected_folder_index]
    elif selected_feature == 'nodule-type':
        info_text = f"Nodule type: {type_of_nodule[selected_folder_index]}"
        feature_data = type_of_nodule
        selected_feature_data = type_of_nodule[selected_folder_index]

    # Create a scatter plot of all feature data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(feature_data))),
        y=feature_data,
        mode='lines+markers',
        line=dict(color='gray', dash='dot'),
        marker=dict(size=[30 if x == selected_folder_index else 10 for x in range(len(feature_data))],
                    color=['red' if x == selected_folder_index else 'blue' for x in range(len(feature_data))])
    ))
    fig.update_layout(title=selected_feature, xaxis_title="Folder Index", yaxis_title=selected_feature)

    return info_text, fig

if __name__ == "__main__":
    app.run_server(debug=True, port=80)
