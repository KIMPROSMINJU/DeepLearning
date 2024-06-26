from keras.datasets import mnist
from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf
from PIL import Image

# Load MNIST dataset
(images, labels), (test_images, test_labels) = mnist.load_data()

# Image Preprocessing
def preprocess_images(images):
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images, dtype=tf.float32))
    images = tf.image.resize(images, [32, 32])
    images /= 255.0  # Normalize to [0, 1] range
    return images

train_images = preprocess_images(images[10000:])
val_images = preprocess_images(images[:10000])
test_images = preprocess_images(test_images)
train_labels = keras.utils.to_categorical(labels[10000:])
val_labels = keras.utils.to_categorical(labels[:10000])
test_labels = keras.utils.to_categorical(test_labels)

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),  # Increased rotation range
        layers.RandomZoom(0.3),      # Increased zoom range
        layers.RandomContrast(0.2)   # Added contrast adjustment
    ]
)

# VGG16 model preparation
conv_base = keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

# Model Definition
inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.GlobalAveragePooling2D()(x)  # Replaced Flatten with GlobalAveragePooling2D
x = layers.Dense(512, activation='relu')(x)  # Increased number of units
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Set learning rate and learning rate scheduler
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="finalConv.h5",
        save_best_only=True,
        monitor="val_loss"
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

# Set batch size
batch_size = 128  # Reduced batch size for better gradient updates

history = model.fit(
    train_images, train_labels,
    epochs=50,  # Increased number of epochs
    batch_size=batch_size,
    validation_data=(val_images, val_labels),
    callbacks=callbacks
)

import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'bo', label="Training acc")
plt.plot(epochs, val_accuracy, 'b', label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'bo', label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

test_model = keras.models.load_model("finalConv.h5")
test_loss, test_acc = test_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
