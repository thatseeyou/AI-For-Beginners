#
# Keras를 사용한 MNIST 데이터셋 분용
#
from tensorflow import keras
import matplotlib.pyplot as plt
import pylab

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# x_train과 x_test의 차원 변경
x_train_reshaped = x_train.reshape(60000, 784)
x_test_reshaped = x_test.reshape(10000, 784)

# 변경된 차원 확인
assert x_train_reshaped.shape == (60000, 784)
assert x_test_reshaped.shape == (10000, 784)

show_data = False
if show_data:
    # 첫 번째 data의 내용을 보기 좋게 출력한다.
    matrix = x_train_reshaped[0].reshape(28, 28) 
    for i in range(matrix.shape[0]):
        print(' '.join(format(x, '3') for x in matrix[i, :28]))

    fig = pylab.figure(figsize=(10,5))
    for i in range(10):
        ax = fig.add_subplot(1,10,i+1)
        pylab.imshow(x_train_reshaped[i].reshape(28,28), cmap="gray")
    pylab.show()
    plt.show()

train_x_norm = x_train_reshaped / 255
test_x_norm = x_test_reshaped / 255

model = keras.models.Sequential([
    # keras.layers.Dense(2352,input_shape=(28*28,),activation='relu'),
    keras.layers.Dense(2352,input_shape=(28*28,),activation='tanh'),
    keras.layers.Dense(10,activation='softmax')
])
model.compile(keras.optimizers.Adam(0.01),'sparse_categorical_crossentropy',['acc'])

hist = model.fit(x=train_x_norm,y=y_train,
                 validation_data=[test_x_norm,y_test],batch_size=210,epochs=10)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.show()