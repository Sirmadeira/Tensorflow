import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train=x_train.astype('float32') / 255.0
x_test=x_test.astype('float32') / 255.0

model=keras.Sequential(
	[
	keras.Input(shape=(32,32,3)),
	layers.Conv2D(32,3,padding='valid',activation='relu'),
	#Executando processando de convulucao 
	layers.MaxPooling2D(),
	#Local onde o filtro, representado pelo seu numero no feature map fez o melhor trabalho
	layers.Conv2D(64,3,activation='relu'),
	layers.MaxPooling2D(),
	#Repeticao do processo
	layers.Conv2D(128,3,activation='relu'),
	layers.Flatten(),
	#Esmaga o input, basicamente deixa num shape so. Voce esmaga para poder passsar por um node tipico
	layers.Dense(64,activation='relu'),
	layers.Dense(10)
	]
)
# print(model.summary())

#Functional
def meu_modelo():
	inputs = keras.Input(shape=(32, 32, 3))
	x = layers.Conv2D(32, 3,padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
	#Acrescentanto regularizer para diminuir o overfitting entre a data treinada e a data testada
	x = layers.BatchNormalization()(x)
	x = keras.activations.relu(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(64, 3,padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = keras.activations.relu(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Conv2D(128, 3,padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	#Batch normalization serve para melhorar a situacao dela 'normalizando' ver video teorico
	x = keras.activations.relu(x)
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(10)(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model

model=meu_modelo()

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=keras.optimizers.Adam(lr=3e-4),
	metrics=['accuracy']
	)

model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2)
model.evaluate(x_test,y_test, batch_size=64,verbose=2)