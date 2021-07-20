import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test)=mnist.load_data()
#Carregando o dataset mnist, akl dataset de letras curvadas
x_train= x_train.reshape(-1,28*28).astype('float32') / 255.0
x_test=x_test.reshape(-1,28*28).astype('float32') /  255.0
#Danda uma chapada no numero de dimensoes para ser, carregando elas como float 32 para diminuir computacao e dividindo por 255
#Para por os valores entre 0 e 1 para diminuir a computacao

#Sequential API (Conveniente, nao e flexivel, so deixa mapear um input layer e um output ae quebra os joelhos)
model=keras.Sequential(
	[
	keras.Input(shape=(28*28)),
	layers.Dense(512,activation='relu'),
	layers.Dense(256, activation='relu'),
	layers.Dense(10),
	]
	)
#Contruindo o modelo da neural network, a funcao de ativacao como voce pode ver e a relu
#O ultima layer e output logo ele n tem activation function
# print(model.summary())
#Resumo do modelo contruido, para se utilizar disso nao esqueca de destacar o shape do input que senao so da para por pos fit

model=keras.Sequential()
model.add(keras.Input(shape=784))
model.add(layers.Dense(512,activation='relu'))
#print(model.summary())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(10))
#Esse tipo de modelo e um pouquinho diferente mas extremamente util para debuging
#Como voce pode ver eu posso verifica dentro de cada layer 


#Functional API(Mais flexivel)

inputs=keras.Input(shape=(784))
x=layers.Dense(512,activation='relu',name='primeiro_layer')(inputs)
x=layers.Dense(256,activation='relu',name='segundo_layer')(x)
outputs=layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)#Garantindo construcao do modelo na functional

#Como se pode ver nesse api eu posso nomear os meus layers
print(model.summary())
#E quando eu dou o summary ele me mostra o nome de cada

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
	#from_logits= true, ira mandar para uma softmax layer, primeirao.
	#E depois vai mapear(analisar os losses), atraves de uma sparsecategoricalcrossentropy
	#Sparse caregorical, e utilizada quando voce tem mais de ou dois labels(y_train) a gente espera labels a serem providos como integers
	#One hot- voce usa	CategoricalCrossEntropy, pq so tem um label
	optimizer=keras.optimizers.Adam(learning_rate=0.001),
	#Adam optimizer famoso optimize  aviso pesquisar
	metrics=['accuracy'],
	)
#Compilando seria, especifica a loss function quais optimizers

model.fit(x_train,y_train,batch_size=32, epochs=5, verbose=2)
#Batch size, quantidade que passa por, epochs quantidade de vezes que passa por input e output, verbose numero de vezes que e printado
model.evaluate(x_test,y_test,batch_size=32,verbose=2)