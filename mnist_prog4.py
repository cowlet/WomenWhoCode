import sys
import csv
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

if len(sys.argv) != 2:
    print("No inut file specified. Usage: \n\tpython3 mnist_prog4.py <inputfile.csv>")
    sys.exit(0)

image = []
with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    for row in reader:
        image.append([int(item) for item in row])

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
img = np.array(image).reshape(1, 784)

# Rescale the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
img = img.astype('float32')
x_train /= 255
x_test /= 255
img /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Set the network architecture
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(784,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Build the network
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train the network
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=4,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

img_pred = model.predict(img)
print("Predicting the input file is:", img_pred.argmax())

