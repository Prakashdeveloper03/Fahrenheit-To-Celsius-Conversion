import numpy as np
import tensorflow as tf

# Define the training data
c = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
f = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with specified loss function, optimizer and evaluation metric
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mean_squared_error'])

# Train the model on the training data for specified number of epochs
model.fit(c, f, epochs=500, verbose=False)
print("Finished training the model")

# Test the trained model
celsius = float(input("Enter the Celsius : "))
print(model.predict([celsius]))

# Calculate the Fahrenheit temperature by hand and compare with the model's prediction
fahrenheit_by_hand = (celsius * 1.8) + 32
print(fahrenheit_by_hand)