# model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, DenseNet169
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = Input(shape=input_shape, name="main_input")
    resnet_base = ResNet50V2(weights='imagenet', include_top=False, pooling='avg', name='resnet_base')
    densenet_base = DenseNet169(weights='imagenet', include_top=False, pooling='avg', name='densenet_base')

    resnet_out = resnet_base(inputs)
    densenet_out = densenet_base(inputs)

    combined = Concatenate(name='concatenate')([resnet_out, densenet_out])

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='fc1')(combined)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)

    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=output, name='AlphaHybridModel')
    return model
