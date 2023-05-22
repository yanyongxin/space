'''
Created on May 14, 2023

@author: yanyo
'''

from datetime import datetime
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


start_time = datetime.now()
print("start_time:", start_time)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

print("Training Complete, now testing: ")
model.fit(x_test, y_test, epochs=1)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Forward pass through the MLP
predicted_labels = model.predict(x_test)

# Convert predicted labels to class predictions
predicted_classes = np.argmax(predicted_labels, axis=1)

# Compare predicted classes with true classes and calculate accuracy
accuracy = np.mean(predicted_classes == y_test)

print("Accuracy on training set:", accuracy)

# Compute the cross-entropy loss
loss = -np.mean(np.log(predicted_labels[np.arange(len(y_test)), y_test]))

print("Cross-entropy loss on training set:", loss)

#probability_model(x_test[:5])
end_time = datetime.now()
print("end_time:", end_time)
time_difference = (end_time - start_time).total_seconds()
print("Duration in seconds:", time_difference)

