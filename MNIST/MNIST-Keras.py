#
# Keras를 사용한 MNIST 데이터셋 응용
#
import gzip
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import pylab

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# assert x_train.shape == (60000, 28, 28)
# assert x_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,)
# assert y_test.shape == (10000,)

# # x_train과 x_test의 차원 변경
# x_train_reshaped = x_train.reshape(60000, 784)
# x_test_reshaped = x_test.reshape(10000, 784)

# # 변경된 차원 확인
# assert x_train_reshaped.shape == (60000, 784)
# assert x_test_reshaped.shape == (10000, 784)

#
# 1. 데이터를 준비한다.
with gzip.open('../lessons/data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

data = MNIST['Train']['Features']
labels = MNIST['Train']['Labels']
assert data.shape == (42000, 784)
assert labels.shape == (42000,)

# 일부 데이터(20%)는 검증용으로 사용한다.
features_train, features_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.2)
assert features_train.shape == (33600, 784)
assert labels_train.shape == (33600, )
assert features_test.shape == (8400, 784)
assert labels_test.shape == (8400, )

show_data = False
if show_data:
    # 첫 번째 data의 내용을 보기 좋게 출력한다.
    matrix = features_train[0].reshape(28, 28) 
    for i in range(matrix.shape[0]):
        print(' '.join(format(x, '3') for x in matrix[i, :28]))

    fig = pylab.figure(figsize=(10,5))
    for i in range(10):
        ax = fig.add_subplot(1,10,i+1)
        pylab.imshow(features_train[i].reshape(28,28), cmap="gray")
    pylab.show()
    plt.show()

# 0 ~ 1의 범위로 정규화
features_train_norm = features_train / 255.0
features_test_norm = features_test / 255.0

model = keras.models.Sequential([
    # keras.layers.Dense(2352,input_shape=(28*28,),activation='relu'),
    # output 2352, input 784
    keras.layers.Dense(2352,input_shape=(28*28,),activation='tanh'),
    # output 10, input 2352
    keras.layers.Dense(10,activation='softmax')
])
# model.compile(keras.optimizers.Adam(0.01),'sparse_categorical_crossentropy',['acc'])
model.compile(keras.optimizers.SGD(0.035),'sparse_categorical_crossentropy',['acc'])

hist = model.fit(x=features_train_norm,y=labels_train,
                 validation_data=[features_test_norm,labels_test],
                 batch_size=210,
                 epochs=10)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.show()