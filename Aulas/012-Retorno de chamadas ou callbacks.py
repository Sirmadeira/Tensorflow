import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds



physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Callbacks seria uma maneira de custumizar o seu modelo, durante o treinamento e o testing
#A gente vai aprender a salvar modulus depois de epochs, e usar learning rate scheduler.
#Eu ja ensinei como salvar no final do treinamento agora vamo aprender no meio

(ds_train,ds_test),ds_info=tfds.load(
	"mnist",
	split=["train", "test"],
    shuffle_files=True,
    as_supervised=True, 
    with_info=True, )


def normalize_img(image, label):
    """Normaliza imagens"""
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128


# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

model=keras.Sequential([
	keras.Input((28,28,1)),
	layers.Conv2D(32,3,activation='relu'),
	layers.Flatten(),
	layers.Dense(10,activation="softmax")])

#Isso daki vai salvar  os pesos e arquivar numa file chamado checkpoint
save_callback=keras.callbacks.ModelCheckpoint(
	'checkpoint/',
	save_weights_only=True,
	monitor='accuracy',
	save_best_only=False,
	)
#Um learning rate scheduler seria algo utilizado, para quando estiver no meio da epoch ele atualizar para um learning rate menor
#Tem muito tipos de callbacks olha no tensorflow, https://www.tensorflow.org/guide/keras/custom_callback
def scheduler(epoch,lr):
	if lr <2:
		return lr
	else:
		return lr*0.99

lr_scheduler=keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
class CallbackCustom(keras.callbacks.Callback):
	def on_epoch_end(self,epoch,logs=None):
		print(logs.keys())
		#Isso daki printa as infos de cada epoch 
		if logs.get("accuracy")>0.90:
			print("Precisao acimade 90 porcento")
			self.model.stop_training =True

model.compile(
	optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
    )

model.fit(ds_train, epochs=10, verbose=2,callbacks=[save_callback,lr_scheduler,CallbackCustom()])
#Lembrese callbacks tem que estar em uma listamodel.fit(ds_train, epochs=5, verbose=2)
model.evaluate(ds_test)