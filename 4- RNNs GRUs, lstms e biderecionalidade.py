import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test)=mnist.load_data()


x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
