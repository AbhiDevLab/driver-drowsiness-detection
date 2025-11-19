"""
Train a simple CNN to classify eye images as open/closed.
Expected dataset layout:
 dataset/open_eyes/*.jpg
 dataset/closed_eyes/*.jpg

Saves model to ../models/cnn_model.h5
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def build_model(input_shape=(64,64,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train(dataset_dir='..\dataset', model_out='../models/cnn_model.h5', epochs=15, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                       rotation_range=10, horizontal_flip=True)
    train_gen = train_datagen.flow_from_directory(dataset_dir,
                                                  target_size=(64,64),
                                                  batch_size=batch_size,
                                                  class_mode='binary', subset='training')
    val_gen = train_datagen.flow_from_directory(dataset_dir,
                                                target_size=(64,64),
                                                batch_size=batch_size,
                                                class_mode='binary', subset='validation')
    model = build_model((64,64,3))
    cb = [ModelCheckpoint(model_out, save_best_only=True), EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cb)
    # save plots
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    plt.figure()
    plt.plot(hist.history['accuracy'], label='acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig('outputs/accuracy_plot.png')
    plt.figure()
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig('outputs/loss_plot.png')
    print('Training complete. Model saved to', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--out', default='models/cnn_model.h5')
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    train(dataset_dir=args.dataset, model_out=args.out, epochs=args.epochs)
