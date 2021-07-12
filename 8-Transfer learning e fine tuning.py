  
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#========================================#
#Carregando um modelo meu pre treinado#
#========================================#
#Modelo pre-treinado
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28* 28).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28* 28).astype("float32") / 255.0

# model=keras.models.load_model('pretreinado/')
# model.trainable= False
# #Isso daki congela os layers, de treinamento
# for layer in model.layers:
# 	assert layer.trainable == False
# 	#Isso daki faz todos deles falso igual ao de cima
# 	layer.trainable==False
# 	#Mas voce pode individualizar alguns


# #Nao rode nao vai funfar
# base_inputs=model.layers([0].input)
# #Selecionando o primeiro layer, nesse caso e imaginativo
# base_outputs=model.layers([-2].output)
# #Selecionando o penultimo layer, ignorando o ultimo dense layer imaginativo
# final_outputs=layers.Dense(10)(base_outputs)
# #Redefinindo o ultimo layer
# novo_model=keras.Model(inputs=base_inputs,outputs=final_outputs)

# novo_model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

# novo_model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
#========================================#
#Carregando um modelo  pre treinado do KERAS#
#========================================#
x= tf.random.normal(shape=(5,299,299,3))
#A data e um conjunto de imagens, que estao em 299 pixes por 299, com 3 canais de cores nesse caso rgb 
y=tf.constant([0,1,2,3,4])

model=keras.applications.InceptionV3(include_top=True)
#A e carregando o modelo
#Para os ultimos totalme conectados layers, a gente usa esse include top para nao remove-los. 
#Caso removeseemo pegariamos uma serie de feature vectors e aproveitando eles
base_inputs=model.layers[0].input
base_outputs=model.layers[-2].output
final_outputs=layers.Dense(5)(base_outputs)
novo_modelo=keras.Model(inputs=base_inputs,outputs=base_outputs)
novo_modelo.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

novo_modelo.fit(x, y, epochs=15, verbose=2)

#========================================#
#Carregando um modelo  pre treinado do tensorflow hub#
#========================================#

x= tf.random.normal(shape=(5,299,299,3))
y = tf.constant([0, 1, 2])

url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
#Url obtido do site do tensorflow hub, mesmo inception v3

base_model=hub.KerasLayer(url,input_shape=(299,299,3))
base_model.trainable=False

model=keras.Sequential([
	layers.Dense(128,activation='relu'),
	layers.Dense(64,activation='relu'),
	layers.Dense(5)])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x, y, epochs=15, verbose=2)