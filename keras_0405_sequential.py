from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, activation="softmax", name="department")(features)

model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])

num_samples = 1280
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
              loss={"priority" : "mean_squared_error", "department" : "categorical_crossentropy"},
              metrics={"priority" : ["mean_absolute_error"], "department" : ["accuracy"]})
model.fit({"title" : title_data, "text_body" : text_body_data, "tags" : tags_data},
           {"priority" : priority_data, "department" : department_data}, epochs=1)
model.evaluate({"title" : title_data, "text_body" : text_body_data, "tags" : tags_data},
               {"priority" : priority_data, "department" : department_data})
priority_preds, department_preds = model.predict({"title" : title_data, "text_body" : text_body_data, "tags" : tags_data})

# keras.utils.plot_model(model, 'aaa.png')
# keras.utils.plot_model(model, "ticket_classifier_with_shape_info.png", show_shapes=True)