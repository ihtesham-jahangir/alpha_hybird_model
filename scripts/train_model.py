#!/usr/bin/env python

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from alpha_hybird_model import build_model, plot_training_history

def main():
    parser = argparse.ArgumentParser(description='Train the Alpha Hybrid CNN model.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save outputs')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    args = parser.parse_args()

    # Set seeds for reproducibility
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2  # Adjust as needed

    # Data generators
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen_val = ImageDataGenerator(rescale=1./255)

    train_generator = datagen_train.flow_from_directory(
        args.train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    valid_generator = datagen_val.flow_from_directory(
        args.val_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Build and compile the model
    model = build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=valid_generator,
        callbacks=[early_stopping, lr_reduction],
        verbose=1
    )

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'alpha_hybird_model_weights.h5')
    model.save_weights(model_path)
    print(f'Saved model weights to {model_path}')

    # Plot history
    if args.plot:
        plot_training_history(history)

if __name__ == '__main__':
    main()
