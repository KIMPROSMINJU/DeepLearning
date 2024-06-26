from keras.datasets import mnist
from tensorflow import keras
from keras import layers

(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28*28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28*28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features1 = layers.Dense(512, activation="relu")(inputs)
    features2 = layers.Dense(512, activation="relu")(inputs)
    features1 = layers.Dense(128, activation="relu")(features1)
    features2 = layers.Dense(64, activation="relu")(features2)
    features = layers.Concatenate()([features1,features2])
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

epochs = 3

model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels,
        epochs=epochs,
        validation_data=(val_images ,val_labels)
)   
test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)

keras.utils.plot_model(model, "mnist_with_shape_info.png", show_shapes=True)
