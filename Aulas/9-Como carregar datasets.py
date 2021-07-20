import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Agora a gente vai aprender o processo de carregar um dataset, tanto o preprocessamento , o pre angariacao entre outros
#Tensorflow.data e um wrapper API capaz de carregar custom datasets e faz um input pipeline, basicamente uma conexao direta

(ds_train,ds_test),ds_info=tfds.load(
	"mnist",
	split=["train","test"],
	shuffle_files=True,
	as_supervised=True,
	with_info=True,
	)

#figs=tfds.show_examples(ds_train,ds_info,rows=4,cols=4)
#Isso daqui vai demonstra alguns exemplos do dataset
#print(ds_info)
#Ds info para demonstra as infos do dataset
#Shuffle files, server para carregar o tf records, nesse caso eles se utilizam disso, para poder passar pelo servidor
#As superviser retorna uma tupla no formata #(img,label) 

def normalize_img(image,label):
	#Normalizacao do datatype e da info
	return tf.cast(image,tf.float32)/255, label


AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE=64
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
#Esse num of parallel calls = autotune, serve para quando carregarmos o dataset a gente nao tem uma sequencia de heranca, logo isso pode ser feito
#Em paralelo, o autotune e utilizado para o tensorflow avaliar se esta correto ou nao.
ds_train=ds_train.cache()
#Vai armazenar o dataset
ds_train=ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)
#O mesmo tem que ser feito com teste mas nao o shuffling e o cache
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(128)
ds_test=ds_test.prefetch(AUTOTUNE)

model=keras.Sequential([
	keras.Input((28,28,1)),
	layers.Conv2D(32,3,activation='relu'),
	layers.Flatten(),
	layers.Dense(10)])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
model.fit(ds_train,epochs=5,verbose=2)
#Voce ve que aqui eu n precisso dar call em x  e y, porque la em cima ja recebe em tuplas 
model.evaluate(ds_test)