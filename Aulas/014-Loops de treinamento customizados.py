import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

#Como consiguir mais flexibilidade ainda no treinamento

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128


ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)


ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(128)
ds_test = ds_train.prefetch(AUTOTUNE)


model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

num_epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
acc_metric = keras.metrics.SparseCategoricalAccuracy()


#Training loop

#Todo que a gente esta fazendo e removendo o model fit, e nomeando o numero de epochs que a gente qr treinar
for epoch in range(num_epochs):
	print(f'Comeco da epoca de treinamento {epoch}')
	for batch_idx,(x_batch,y_batch) in enumerate(ds_train):
		with tf.GradientTape() as tape:
			y_pred=model(x_batch,training=False)
			loss=loss_fn(y_batch,y_pred)
		gradients=tape.gradient(loss,model.trainable_weights)
		optimizer.apply_gradients(zip(gradients,model.trainable_weights))
		acc_metric.update_state(y_batch,y_pred)
	train_acc=acc_metric.result()
	print(f"Precisao acima da epoch {train_acc}")
	acc_metric.reset_states()

#Loop de teste


for batch_idx,(x_batch,y_batch) in enumerate(ds_test):
	y_pred=model(x_batch,training=True)
	acc_metric.update_state(y_batch,y_pred)

train_acc=acc_metric.result()
print(f"Precisao no teste set {train_acc}")
acc_metric.reset_states()