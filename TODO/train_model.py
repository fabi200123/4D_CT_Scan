import os
import numpy as np
from unet_model import unet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(input_folder):
    all_files = os.listdir(input_folder)
    print(f"All files in folder: {all_files}")

    # Load the data from the .npy files
    data = [np.load(os.path.join(input_folder, f)) for f in all_files]

    # Separate images and masks
    images = [item[:, :, :, 0] for item in data]
    masks = [item[:, :, :, 1] for item in data]

    x = np.stack(images)
    y = np.stack(masks)

    return x, y

def train_unet(input_folder, model_output_path, epochs=100, batch_size=8, gpu_device=1, memory_limit=28):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_device], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_device], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[gpu_device],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)]
    )

    x, y = load_data(input_folder)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = unet(input_size=(512, 512, 1))

    checkpoint = ModelCheckpoint(model_output_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
              callbacks=[checkpoint, early_stopping])

if __name__ == '__main__':
    input_folder = "C:\\Users\\fabi2\\OneDrive\\Desktop\\LUNA16-Dataset\\Output"
    model_output_path = 'unet_luna16_weights.hdf5'
    train_unet(input_folder, model_output_path, gpu_device=1, memory_limit=28)
