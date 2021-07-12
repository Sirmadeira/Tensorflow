import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28* 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28* 28).astype("float32") / 255.0

#Montando o meu layer proprio
class Dense(layers.Layer):
	def __init__(self,units):
		#Aqui antigamente tinha input dim mas vou retirar pq agora nao vai ser mais necessario defini-la
		super().__init__()
		self.units=units
	#Esse build metodo tem como funcao, tornar os layer preguicosos, ou seja, n precisa nem definir a input dim 
	def build(self,input_shape):
		self.w=self.add_weight(
			name="w",
			shape=(input_shape[-1],self.units),
			#Definindo a formatacao dos dados que entram nesse caso seria a especificada em x train e x test
			initializer='random_normal',
			#Ver teorica para entender initializers, mas basicamente seria a maneira como a gente define os primeiros pesos e bias
			trainable=True,
			#Garantindo que todos os labels sao treinavei
			)
		#Criando o bias
		self.b=self.add_weight(
			name='b',shape=(self.units,),initializer='zeros',trainable=True,
			)

	def call(self,inputs):
		return tf.matmul(inputs,self.w)+self.b
		#Multiplicando matrizes
#Definindo meu proprio relu
class MeuRelu(layers.Layer):
	def __init__(self):
		super().__init__()

	def call(self,x):
		return tf.math.maximum(x,0)


#Criando custom models, para ter um maior poder de customizacao sobre os nodes

class MeuModelo(keras.Model):
	def __init__(self,num_classes=10):
		super(MeuModelo,self).__init__()
		self.dense1= Dense(64)
		#Para evitar o tipo de dimensao logo em seguida na call, a gente se utiliza da build aqui
		self.dense2= Dense(num_classes)
		self.relu=MeuRelu()

	def call(self,input_tensor):
		x=self.relu(self.dense1(input_tensor))
		return self.dense2(x)


model= MeuModelo()

model.compile(
	optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],)

model.fit(x_train,y_train,batch_size=32,epochs=3,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)