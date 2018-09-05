import random
import numpy as np
from matplotlib import pylab as plt

from keras.datasets import mnist
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import model_from_yaml

# Define number of classes
num_classes = 10
# Input (image) dimensions
rows, cols = 28, 28

# mnist.load_data() returns 2 tuples split into training/testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display image to test
image_num = random.randint(0, 10000)
plt.imshow(x_test[image_num], cmap=plt.get_cmap('gray'))
plt.show()

# If color channels are last parameter
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1).astype('float32')
y_test = to_categorical(y_test, num_classes)

# Normalize pixel values between 0 and 1 per channel
x_test /= 255 

test_image = x_test[image_num,:,:,:]
test_image = np.expand_dims(test_image, axis=0)

test_label = y_train[0]
test_label = to_categorical(test_label, num_classes)

# ----------------------------------------------------

with open('model.yaml', 'r') as yaml_file:
	loaded_yaml_model = yaml_file.read()

# Create model from yaml config
loaded_model = model_from_yaml(loaded_yaml_model)

# Load weights from trained network
loaded_model.load_weights('model_weights.h5')
print("Loaded model and weights from disk")

prediction = loaded_model.predict(test_image, batch_size=1, verbose=1)
print('Prediction: {} with {:.2f}% confidence.'.format(np.argmax(prediction), np.max(prediction)*100))
