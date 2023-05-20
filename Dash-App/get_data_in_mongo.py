import os
import numpy as np
import pymongo
from modules.nodule_features import get_all_features
from pymongo import MongoClient
from bson import ObjectId


# Your MongoDB Atlas cluster connection string
MONGO_CONNECTION_STRING = "mongodb+srv://dianavelciov:parola@cluster0.qqmezlq.mongodb.net/cool_notes_app?retryWrites=true&w=majority"

# Create a MongoClient to the running MongoDB Atlas cluster instance
client = MongoClient(MONGO_CONNECTION_STRING)

# Getting a Database
db = client.cool_notes_app

# Getting a Collection
collection = db.patients

def get_subdirectories(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

path_to_data = "C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\"
name_of_pacient = "data"
path_to_data += name_of_pacient + "\\"
data_folder = path_to_data + "converted_nrrds\\"
subdirectories = get_subdirectories(data_folder)

# Initialize the vectors for features
nodule_volume = []
nodule_fractal_dimension = []
nodule_area = []
calcification = []
spiculation = []
type_of_nodule = []

nodule_volume, nodule_fractal_dimension, nodule_area, calcification, spiculation, type_of_nodule = get_all_features(data_folder, subdirectories) 

data = []
for i, selected_folder in enumerate(subdirectories):
    data.append({
        "nodule_volume": nodule_volume[i],
        "nodule_area": nodule_area[i],
        "fractal_dimension": nodule_fractal_dimension[i],
        "calcification": calcification[i].tolist() if isinstance(calcification[i], np.ndarray) else calcification[i],
        "spiculation": spiculation[i].tolist() if isinstance(spiculation[i], np.ndarray) else spiculation[i],
        "type_of_nodule": type_of_nodule[i]
    })
    print("Data is:", data[i])

pacient_id = ObjectId('645430a43b0ec4b7df36aec6')

result = collection.update_one({"_id": pacient_id}, {"$set": {"Data": data}}, upsert=True)
print("Matched documents: ", result.matched_count)
print("Modified documents: ", result.modified_count)
doc = collection.find_one({"_id": pacient_id})
print(doc)

