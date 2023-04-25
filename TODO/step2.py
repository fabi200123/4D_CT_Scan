import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from unet_3d import unet_3d
from label_nodules import label_nodules

# Load your preprocessed data (X and y) here
input_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\Output"
npy_files = glob.glob(os.path.join(input_folder, "*.npy"))

# Combine the preprocessed data into a single NumPy array and create the corresponding labels
X = []
y = []  # Replace this with the actual nodule labels (1 for nodules, 0 for non-nodules)

for npy_file in npy_files:
    preprocessed_data = np.load(npy_file)
    X.append(preprocessed_data)

X = np.array(X)
y = label_nodules()

# Make sure X has the shape (num_samples, 64, 64, 64, 1)
X = np.expand_dims(X, axis=-1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the model
model = unet_3d()
model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
epochs = 100
batch_size = 8
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# Save the trained model
model.save("3d_unet_model.h5")
