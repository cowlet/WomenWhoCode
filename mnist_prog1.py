from keras.datasets import mnist

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

