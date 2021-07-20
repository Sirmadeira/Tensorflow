import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,28,28,1).astype('float32') / 255.0
x_test=x_test.reshape(-1,28,28,1).astype('float32') / 255.0

#Existe muitos cenarios em que voce precisa modificar o jeito como os training steps estao sendo feitos no model.fit
#A gente vai aprender como construir um nosso mesmo, e se utilizar dele nesse modle.fit

model=keras.Sequential(
	[
		layers.Input(shape=(28,28,1)),
		layers.Conv2D(64,3,padding='same'),
		layers.ReLU(),
		layers.Conv2D(128,3,padding='same'),
		layers.ReLU(),
		layers.Flatten(),
		layers.Dense(10),
	],
	name='model',
	)

#Funcao que faz o treinamento
class CustomFit(keras.Model):
	def __init__(self,model):
		super(CustomFit, self).__init__()
		self.model=model


	def compile(self,optimizer,loss):
		super(CustomFit, self).compile()
		self.optimizer=optimizer
		self.loss = loss

	def train_step(self,data):
		#Abre a data
		x,y=data
		with tf.GradientTape() as tape:
			#Agora a gente vai fazer a proprogacao forward basicamenter a normal, e a loss function
			#E quando a gente fizer debaixo dessa tape, ele vai memorizar todos as operacoes o que sera util quando calcularmos
			#Os gradientes na backpropagation
			#Computa predicoes
			y_pred=self.model(x,training=True)
			#Atualiza as metricas seguindo a loss
			loss=self.loss(y,y_pred)
		training_vars=self.trainable_variables
		gradients=tape.gradient(loss,training_vars)

		self.optimizer.apply_gradients(zip(gradients,training_vars))
		acc_metric.update_state(y,y_pred)

		return {'loss':loss,"accuracy":acc_metric.result()}
		#m. name seria as variaveis exemplo loss
		#Depois a gente pega o result que e a loss atual
		#E a gente fazisso para todos a metricas que nesse caso seria a loss e a precisaso
	def test_step(self,data):
		x,y=data
		y_pred=self,model(x,training=False)
		loss=self.loss(y,y_pred)
		acc_metric.update_state=(y,y_pred)
		return {'loss':loss,"accuracy":acc_metric.result()}
		

acc_metric=keras.metrics.SparseCategoricalAccuracy(name="accuracy")
training=CustomFit(model)
training.compile(
	optimizer=keras.optimizers.Adam(learning_rate=3e-4),
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	)


training.fit(x_train,y_train,batch_size=64,epochs=2)
training.evaluate(x_test,y_test,batch_size=64)