# code based on https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

import numpy as np
from tensorflow import expand_dims, keras
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = expand_dims(x_train / 255, 3)
x_test  = expand_dims(x_test / 255, 3)

x_valid, y_valid = x_train[:5000], y_train[:5000]

lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(28, 28, 1), padding='same'),
    keras.layers.AveragePooling2D(),
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
    keras.layers.AveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='tanh'),
    keras.layers.Dense(84, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])

lenet_5_model.compile(optimizer='sgd', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

lenet_5_model.fit( x_train, y_train, epochs=5, validation_data=(x_valid, y_valid) )

# lenet_5_model.evaluate(x_test, y_test)

y_hat = lenet_5_model.predict(x_test)
y_pred = np.argmax(y_hat, axis=1)
conf_matrix = confusion_matrix( y_test, y_pred )

print( conf_matrix )
