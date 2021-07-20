import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Aqui a gente vai aprender a carregar e interpretar linhas de texto
#Por exemplo, esse filme foi terrivel -> associado a 0
#Esse filme foi muito bom ->1

(ds_train,ds_test),ds_info=tfds.load(
	"imdb_reviews",
	split=["train",'test'],
	shuffle_files=True,
	as_supervised=True,
	with_info=True)

#A primeira coisa que a gente ttem que fazer e tokenizar. Tokenizar e separar os valores da string, e por elas no mesmo nivel
#Ele basicamente e uma lista, de strings. ["eu","amo","isso"]

tokenizer=tfds.deprecated.text.Tokenizer()

def construir_vocabulario():
	vocabulario=set()
	for text, _ in ds_train:
		vocabulario.update(tokenizer.tokenize(text.numpy().lower()))
	return vocabulario
#Isso dak e bem simples, ele so adicionaa as palavras. Existe vias mais complexos, por exemplo se uma palavra aparece multiplas  vezes
vocabulario=construir_vocabulario()
encoder=tfds.deprecated.text.TokenTextEncoder(
	vocabulario,oov_token="<UNK>",lowercase=True,tokenizer=tokenizer)
#Encodando para poder passar

def meu_encoding(text_tensor ,label):
	return encoder.encode(text_tensor.numpy()),label

def encode_map(text,label):
	encoded_text,label=tf.py_function(
		meu_encoding,inp=[text,label],Tout=(tf.int64,tf.int64))
	#A py function nao define o formato dos tensores, por isso o tout
#Formatando o mapa do encode
	#tf.data.dataset, funciona melhor se voce definir a shape
	encoded_text.set_shape([None])
	label.set_shape([])
	return encoded_text, label

AUTOTUNE=tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(encode_map,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(10000)
ds_train=ds_train.padded_batch(32,padded_shapes=([None],()))
#A gente precisa esmagar para pegar o tipo de dataset
ds_train=ds_train.prefetch(AUTOTUNE)

ds_test=ds_test.map(encode_map)
ds_test=ds_test.padded_batch(32,padded_shapes=([None],()))

model=keras.Sequential([
	layers.Masking(mask_value=0),
	layers.Embedding(input_dim=len(vocabulario)+2,output_dim=32),
	layers.GlobalAveragePooling1D(),
	layers.Dense(64,activation='relu'),
	layers.Dense(1)#Menor que 0 negativo, maior que 1 positivo
	])
#Masking ignora os valores 0 porque eles sao inuteis
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)


model.fit(ds_train,epochs=10,verbose=2)
#Voce ve que aqui eu n precisso dar call em x  e y, porque la em cima ja recebe em tuplas 
model.evaluate(ds_test)

#isso daki e so uma demonstracao de como pegar um texto de dataset