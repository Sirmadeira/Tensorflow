import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test)=mnist.load_data()


x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


model = keras.Sequential()
model.add(keras.Input(shape=(None,28)))
model.add(
	layers.Bidirectional(
		layers.SimpleRNN(256, return_sequences=True,activation='relu')
		)
	)
#Biderectional como discutid na teorica, e repetir o numero de nodes so que ao contrario
#Para ativar  uma funcao recorrent gru ou lstm e so trocar o nome SIMPLERNN POR GRU LSTM
#Return sequences,monta um timestep seria o loop interno da data por um node, o que faz dela recurrent. Nesse caso o loop e 28 vezes
model.add(layers.SimpleRNN(256,activation='relu'))
#Nesse caso nao nomeie return sequences, logo ela somente faz um timestep com os inputs anteriores, sem rotacionar eles
model.add(layers.Dense(10))
model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=keras.optimizers.Adam(lr=0.001),
	metrics=['accuracy']
	)
model.fit(x_train, y_train,batch_size=64,epochs=10,verbose=2)
model.evaluate(x_test.astype,batch_size=64,verbose=2)