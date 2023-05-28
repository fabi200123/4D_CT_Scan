import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes
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
from flask import Flask, redirect
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(int(timestamp)/1000.0).strftime('%Y-%m-%d')

def get_subdirectories(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def return_fig(images, threshold, step_size):
    p = images.transpose(2, 1, 0)
    verts, faces, _, _ = marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    x, y, z = zip(*verts)
    colormap = ['rgb(255, 192, 203)', 'rgb(236, 236, 212)']
<<<<<<< Updated upstream
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap=colormap, simplices=faces, backgroundcolor='rgb(125, 125, 125)', title="3D Visualization of the CT Scan")
=======
    fig = FF.create_trisurf(x=x, y=y, z=z, plot_edges=False, colormap=colormap, simplices=faces, backgroundcolor='rgb(125, 125, 125)', title="3D Visualization of the Nodule")
>>>>>>> Stashed changes
    fig.layout.scene.xaxis.title = 'Width'
    fig.layout.scene.yaxis.title = 'Height'
    fig.layout.scene.zaxis.title = 'Depth (Slice Number)'
    return fig

def get_png_files(folder):
    png_folder = os.path.join(folder + "_GTV-1_mask\\GTV-1_mask")
    png_files = [os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith(".png")]
    png_files.sort()
    return png_files

# Initialize the vectors for features
nodule_volume = []
nodule_fractal_dimension = []
nodule_area = []
calcification = []
spiculation = []
type_of_nodule = []
initial_fig = None

server = Flask(__name__)

app = dash.Dash(__name__, server=server, prevent_initial_callbacks='initial_duplicate', url_base_pathname='/visualize/')

@server.route('/<path_wanted>', methods=['GET'])

def handle_request(path_wanted):

    # Initialize the vectors for features
    global nodule_volume
    global nodule_fractal_dimension
    global nodule_area
    global calcification
    global spiculation
    global type_of_nodule

    global name_of_pacient
    global path_to_data
    global data_folder
    global subdirectories
    global png_folder

    name_of_pacient = path_wanted
    path_to_data = "C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\"
    path_to_data += name_of_pacient + "\\"
    data_folder = path_to_data + "converted_nrrds\\"
    subdirectories = get_subdirectories(data_folder)
    png_folder = path_to_data + "images_quick_check\\"

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

#    nodule_volume, nodule_fractal_dimension, nodule_area, calcification, spiculation, type_of_nodule = get_all_features(data_folder, subdirectories) 
    # Your MongoDB Atlas cluster connection string
    MONGO_CONNECTION_STRING = "mongodb+srv://dianavelciov:parola@cluster0.qqmezlq.mongodb.net/cool_notes_app?retryWrites=true&w=majority"

    # Create a MongoClient to the running MongoDB Atlas cluster instance
    client = MongoClient(MONGO_CONNECTION_STRING)

    # Getting a Database
    db = client.cool_notes_app

    # Getting a Collection
    collection = db.patients
    pacient_id = ObjectId('645430a43b0ec4b7df36aec6')
    doc = collection.find_one({"_id": pacient_id})

    # Make sure nodule features are not already initialized
    nodule_volume = []
    nodule_fractal_dimension = []
    nodule_area = []
    calcification = []
    spiculation = []
    type_of_nodule = []

    # Extract the data from the 'Data' field in the document
    for data in doc['Data']:
        nodule_volume.append(data['nodule_volume'])
        nodule_area.append(data['nodule_area'])
        nodule_fractal_dimension.append(data['fractal_dimension'])
        calcification.append(data['calcification'])
        spiculation.append(data['spiculation'])
        type_of_nodule.append(data['type_of_nodule'])
    app.layout = html.Div(
        children=[
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
                        style={"display": "inline-block", "vertical-align": "top", "margin-left": "200px"}  # Update the style here
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

    # And return a response to the caller
    return app.index()

@app.callback(
    [Output('graph-with-selector', 'figure'),
    Output('info-text', 'children', allow_duplicate=True),
    # Output('info-graph', 'figure', allow_duplicate=True),
    Output('feature-dropdown', 'value'),
    Output('png-slider', 'max'),
    Output('png-viewer', 'src', allow_duplicate=True),
    Output('png-slider', 'value')],
    [Input('folder-selector', 'value')],
)

def update_figure(selected_folder_index):
    updated_fig = initial_fig
    slider_value = 0
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

    return updated_fig, info_display, selected_feature, max_slider_value, png_src, slider_value

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

    timestamps = []
    for time_stamp in subdirectories:
        timestamps.append(timestamp_to_date(time_stamp.replace("_1-1", "")))

    # Create a scatter plot of all feature data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
<<<<<<< Updated upstream
        x=subdirectories,
=======
        x=timestamps,
>>>>>>> Stashed changes
        y=feature_data,
        mode='lines+markers',
        line=dict(color='gray', dash='dot'),
        marker=dict(size=[30 if x == selected_folder_index else 10 for x in range(len(feature_data))],
                    color=['red' if x == selected_folder_index else 'blue' for x in range(len(feature_data))])
    ))
    fig.update_layout(
    title=selected_feature.capitalize().replace("-", " ") + ' Over Time',
    xaxis_title="CT Scan Date",
    yaxis_title=selected_feature.capitalize().replace("-", " ")
    )


    return info_text, fig


if __name__ == "__main__":
    server.run(debug=True,host='192.168.101.18', port=8080)