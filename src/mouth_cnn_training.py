import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_DIR = 'dataset'
MODEL_DIR = 'models'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape=(64,64,1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7,1.3),
        zoom_range=0.1
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        classes=['no_yawn', 'yawn'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        classes=['no_yawn', 'yawn'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 1))

    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    model.save(os.path.join(MODEL_DIR, 'mouth_cnn_model.h5'))

    print("✔ Mouth CNN model saved as mouth_cnn_model.h5")

if __name__ == '__main__':
    main()
