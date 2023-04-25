import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

def create_labels(input_folder, annotations_file, candidates_file):
    # Read the annotations and candidates CSV files
    annotations = pd.read_csv(annotations_file)
    candidates = pd.read_csv(candidates_file)

    # Create a dictionary mapping the 'seriesuid' to the nodule coordinates
    nodules = defaultdict(list)
    for index, row in annotations.iterrows():
        nodules[row['seriesuid']].append((row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']))

    # Create a dictionary mapping the 'seriesuid' to the candidate coordinates and their respective classes
    candidate_dict = defaultdict(list)
    for index, row in candidates.iterrows():
        candidate_dict[row['seriesuid']].append((row['coordX'], row['coordY'], row['coordZ'], row['class']))

    # For each preprocessed sample, find the corresponding 'seriesuid' and create the label
    labels = []
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))

    for npy_file in npy_files:
        seriesuid = os.path.basename(npy_file).replace(".npy", "")
        candidate_list = candidate_dict[seriesuid]
        has_nodule = False

        for candidate in candidate_list:
            if candidate[3] == 1:
                has_nodule = True
                break

        labels.append(has_nodule)

    return labels

def label_nodules():
    input_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\Output"
    annotations_file = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\CSVFILES\\annotations.csv"
    candidates_file = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\CSVFILES\\candidates.csv"
    y = create_labels(input_folder, annotations_file, candidates_file)
    y = np.array(y, dtype=np.int32)
    return y
