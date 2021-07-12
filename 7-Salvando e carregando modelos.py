  
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

#Ao salvarmos um modelo, existe multipals coisa que podemo vir a querer salvar
#Por exemplo podemo querer salvvar o seus pesos o o modelo em si, as configuracoes de treino e os optimizadores e seus estados
#No entanto, ao salvar tambem queremos saber como carregar um modelo inteiro

model1 = keras.Sequential([layers.Dense(64, activation="relu"), layers.Dense(10)])
#Modelo sequencial

inputs = keras.Input(784)
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)
#Modelo funcional


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)



#Modelo subclasse
model3 = MyModel()

# model = keras.models.load_model('pretreinado/')
#Ao salvar um modelo nao se esqueca de retirar o compile, ele nao e mais necessario pq ja foi definido a loss function
#O optimizer e as metrics
# model.load_weights('pesos/')
#Aqui voce carrega os pesos, lembre a folder tem que ser a mesmo em que foi salva
#Aviso voce nao  pode carregar um modelo que se utilize de um api diferente
#Se foi salvo em funcional, tem que ser carregado em funcional e vice versa
model=model1

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
# model.save_weights('pesos/',save_format='h5')
#Essa linha de codigo salva somente os pesos, save format seria a metodologia de com osalvar o modelo
model.save("pretreinado/")
#Esse e o modelo inteiro sendo salvado