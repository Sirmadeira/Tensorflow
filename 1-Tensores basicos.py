#Tensor e um array multidimensional, que tem a habilidade se ser rodado em gpu
#Tensor e uma generalizacao, basicamente um tensor de uma dimensao e um vetor
#De duas e uma matrix, 0 um scalar. E uma generalizacao
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Diminuindo numero de erros disponibilizados 
import tensorflow as tf

#Configs
# physical_devices=tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0],True)
#Caso esteja em gpu. Avaliar e cortar numero de gpus utilizados
#Evitar erros de memoria

#Inicializando tensors
x=tf.constant(4, shape =(1,1),dtype=tf.float32)
x=tf.constant([[1,2,3],[4,5,6]])
#Exemplos manuais
x=tf.ones((3,3))
#Exemplo mais automatico de matrix, como voce pode ver ja gera 3x3 colunas de valores um
x=tf.zeros((2,3))
x=tf.random.normal((3,3),mean=0, stddev=1)#Exemplo de stddev, que geral valores randoms
x=tf.eye(3)#I e  de matriz identidade akl que a diagonal sempre tem valor 1, eye = I
x=tf.random.uniform((1,3),minval=0,maxval=1)
x=tf.range(start=1,limit=10,delta=2)
#Igual a range do python no entanto delta, destaca de quanto em quanto vai
x=tf.cast(x,dtype=tf.float64)
#Altera a classe do valor

print(x)

#Operacoes matematicas
x=tf.constant([1,2,3])
y=tf.constant([9,8,7])

z=tf.add(x,y)
#z=x+y
z=tf.subtract(x,y)
#z=x-y

z=tf.divide(x,y)
#z=x/y

z=tf.multiply(x,y)
#z=x*y

z=tf.tensordot(x,y ,axes=1)
#Multiplica os elementos e o soma depois
z=tf.reduce_sum(x*y,axis=0)
#Multiplica os elementos e depois tira o valor de cada	

x=tf.random.normal((2,3))
y=tf.random.normal((3,4))

z=tf.matmul(x,y)
#Multiplicando matrizes
# print(z)
# z=x @ y
# print(z)
#Indexing
x = tf.constant([0,1,1,2,1,2,3])
print(x[::2])
#Mecanica interessante para pular partes do vetor
indices=tf.constant([0,3])
x_ind = tf.gather(x,indices)
# print(x_ind)

x=tf.constant([[2,1,3],
	[2,1,3],
	[4,1,2]])
# print(x[0,:])
# print(x[0:2,:])
#Depois da virgula seria em quantas dimensoes eu quero fazer a selecao de parte da matriz, caso eu pusesse : e todas, se pusese 0:1 seria so uam e assim vai

#Reshaping

x=tf.range(9)
print(x)

x=tf.reshape(x,(3,3))
print(x)

x=tf.transpose(x,perm=[1,0])
#Transpote vem de transposta basicamente vc gira a matriz, nesse caso so uma vez, caso queria mais e so aumentar o numero de vezes
print(x)
