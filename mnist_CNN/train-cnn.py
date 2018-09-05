import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# A "class" for each digit from 0-9
num_classes = 10
# How many times to run through the data while training
num_epochs = 1
# Input (image) dimensions
img_rows, img_cols = 28, 28

# mnist.load_data() returns 2 tuples split into training/testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Need to reshape data based on backend preferred image format (TF vs Theano)
if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols).astype('float32')
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
	input_shape = (img_rows, img_cols, 1)

# Normalize pixel values between 0 and 1 per color channel for easier training
x_train /= 255
x_test /= 255

# Convert class label values to one-hot vectors [0, 0, ..., 1, 0, ..., 0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Print out sample sizes
print("Training samples:", x_train.shape[0])
print("Test samples:", x_test.shape[0])

# Build model
model = Sequential()
# Convolutional Layer (input shape specified because its the first layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# Pooling Layer (speeds up computation somewhat by decreasing data size and
# 				 allows for more local features to be learned)
model.add(MaxPooling2D((2, 2)))
# speeds up from 1min to 40 seconds but worse accuracy
#model.add(MaxPooling2D((2, 2)))
# doesn't speed up anymore and worse accuracy
#model.add(MaxPooling2D((2, 2)))

# Flattens data into one dimension for fully connected layer to follow
model.add(Flatten())
# Fully Connected Layer
model.add(Dense(128, activation='relu'))
# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Choosing loss function and optimizer for training
model.compile(loss='categorical_crossentropy',
				optimizer='sgd',
				metrics=['accuracy'])

# Fit to training data (like sklearn classifiers)
model.fit(x_train, y_train, 
		  epochs=num_epochs, 
		  validation_data=(x_test, y_test))

# Save metrics of network like accuracy for observation
score = model.evaluate(x_test, y_test, verbose=0)

# Print out results
print("Test loss:", score[0])
print("Test accuracy: {:.2f}%".format(score[1]*100))

# Save model
model_yaml = model.to_yaml()
with open('model.yaml', 'w') as yaml_file:
	yaml_file.write(model_yaml)

# Save weights
model.save_weights('model_weights.h5')
print('Saved model and weights to disk.')
