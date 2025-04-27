# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,DepthwiseConv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras import regularizers,Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

# %%
img_height,img_width=200,200
num_classes=10
input_shape=(img_height,img_width,3)

# %%
train_dir = "/home/varun/all_codes/kasturi_dl/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/train"
val_dir = "/home/varun/all_codes/kasturi_dl/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/valid"
test_dir = "/home/varun/all_codes/kasturi_dl/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/test"

# %%
train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    #validation_split=0.2,
    #subset="training",
    shuffle=True,
    seed=69,
    image_size=(img_height, img_width),
    batch_size=128,  
)
val_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    #validation_split=0.2,
    #subset="validation",
    shuffle=True,
    seed=69,
    image_size=(img_height, img_width),
    batch_size=128,  
)

# Create test dataset
test_ds = image_dataset_from_directory(
    test_dir, image_size=(img_height, img_width),
    batch_size=128,
    labels='inferred',
    label_mode='categorical'
)

# %%
def build_cnn(input_shape=(200, 200, 3), num_classes=10):
    """Builds a simple convolutional neural network model.

    Args:
        input_shape: Tuple representing the input image shape (height, width, channels).
        num_classes: Integer representing the number of output classes.

    Returns:
        A Keras model instance.
    """

    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  # Start with input layer
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling after first conv

    model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Second set of Conv and Pool
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu')) # Third set of Conv and Pool
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten before dense layers
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization

    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax

    return model

# %%
model = build_cnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
hist = model.fit(train_ds, epochs=15,validation_data = val_ds)

# %%
loss,acc = model.evaluate(test_ds)

# %%
import matplotlib.pyplot as plt
fig2 = plt.figure(figsize=(10, 6))

plt.plot(hist.history['val_accuracy'], color='cyan', label='val accuracy')
plt.plot(hist.history['accuracy'], 'green', label='accuracy')

 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.savefig("/home/hpc/acc.png")

# %%
import matplotlib.pyplot as plt
fig2 = plt.figure(figsize=(10, 6))

plt.plot(hist.history['val_loss'], color='orange', label='val loss')
plt.plot(hist.history['loss'], 'red', label='loss')

 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.savefig("/home/hpc/loss.png")
