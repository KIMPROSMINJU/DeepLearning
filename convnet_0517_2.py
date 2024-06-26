from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


inputs = keras.Input(shape=(6, 5, 1))
x = layers.Conv2D(filters=32, kernel_size=2, activation="relu")(inputs)
# x = layers.MaxPooling2D(pool_size=(1, 2))(x)
x = layers.Conv2D(filters=64, kernel_size=2, activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=32, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(1, 2))(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

data = load_breast_cancer(as_frame=True)
X_train,X_test,y_train,y_test = train_test_split(data.data, data.target,
                                               test_size=0.3, random_state=4321)

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
X_test_std = scalerX.transform(X_test)

X_train_std = X_train_std.reshape((-1, 6, 5, 1))
X_test_std = X_test_std.reshape((-1, 6, 5, 1))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X_train_std, y_train, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(X_test_std, y_test)
print("모델 정확도 : " + str(test_acc))