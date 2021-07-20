import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True, 
    with_info=True,  # Capaz de obter info do dicionario
)

#Data augmentation, seria o processo de aumentar o seu dataset de maneira artificial. No entanto, voce nao aumenta ele diretamente
#Voce aumenta ele no decorrer do processo
#Existe duas maneira de augmentar, a sua data
#A primeira e criando um tf dataset e aplicando a map augment em cada imagen
#A segunda e inserindo no seu modelo


def normalize_img(image, label):
    """Normaliza imagens"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def augment(image,label):
    new_height=new_width=32
    image=tf.image.resize(image,(new_height,new_width))
    #Ranodm resizing
    if tf.random.uniform((),minval=0,maxval=1)<0.1:
        #Isso daki so seria a chance de converter
        image=tf.tile(tf.image.rgb_to_grayscale(image),[1,1,3])
        #Quando a gente converte para greyscale lembrese no numero de canais
        #No final depois de converte a imagen a gente nao esta copiando a primeira dimensao nem a segunda e na ultima a gente copia 3 vezes
        #Isso vai evitar erro no final
    image = tf.image.random_brightness(image,max_delta=0.1)
    image = tf.image.random_contrast(image,lower=0.1,upper=0.2)

    image=tf.image.random_flip_left_right(image)
    return image,label


# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
#ds_train = ds_train.map(augment,num_parallel_calls=AUTOTUNE)
#Aviso esse e o primeiro metodo, que e dar call na func
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

#Segundo metodo, inserindo n omodel odiretamente
data_augmentation=keras.Sequential([
    layers.experimental.preprocessing.Resizing(height=32,width=32),
    layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    layers.experimental.preprocessing.RandomContrast(factor=0.1)
    ])
model = keras.Sequential(
    [
        data_augmentation,
        keras.Input((32, 32, 3)),
        layers.Conv2D(4, 3, padding="same", activation="relu"),
        layers.Conv2D(8, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)



model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=5, verbose=2)
model.evaluate(ds_test)