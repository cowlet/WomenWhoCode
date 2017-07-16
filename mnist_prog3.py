import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("The data starts with shape", x_train.shape, "and", y_train.shape)

# Reshape the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
print("The data becomes shaped as", x_train.shape, "and", y_train.shape)

# Rescale the data
print("The max value in the training set is", x_train.max())
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("After scaling, the max value in the training set is", x_train.max())

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

y_pred = model.predict(x_test)
print(y_test[0])
print(y_pred[0])
print("Actual value:", y_test[0].argmax(), " Predicted:", y_pred[0].argmax())

