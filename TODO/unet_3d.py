from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model

def unet_3d(input_shape=(64, 64, 64, 1)):
    inputs = Input(input_shape)

    # Encoding path
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Decoding path
    up3 = concatenate([UpSampling3D(size=(2, 2, 2))(conv2), conv1], axis=-1)
    conv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up3)
    conv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv3)

    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)

    output = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)

    return Model(inputs=inputs, outputs=output)
