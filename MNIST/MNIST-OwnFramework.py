#
# Framework를 사용하지 않고 MNIST 문제를 푼다.
#
import sys
import numpy as np
import gzip
import pickle
from sklearn.model_selection import train_test_split

#
# 1. 데이터를 준비한다.
with gzip.open('../lessons/data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

data = MNIST['Train']['Features']
labels = MNIST['Train']['Labels']
assert data.shape == (42000, 784)
assert labels.shape == (42000,)

# 일부 데이터(20%)는 검증용으로 사용한다.
train_features, test_features, train_labels, test_labels = train_test_split(data,labels,test_size=0.2)
assert train_features.shape == (33600, 784)
assert train_labels.shape == (33600, )
assert test_features.shape == (8400, 784)
assert test_labels.shape == (8400, )

# 0 ~ 1의 범위로 정규화
train_features_norm = train_features / 255.0
test_features_norm = test_features / 255.0

#
# 첫 번째 data의 내용을 보기 좋게 출력한다.
matrix = train_features[0].reshape(28, 28) 
for i in range(matrix.shape[0]):
    print(' '.join(format(x, '3') for x in matrix[i, :28]))
print(train_labels[0])

#
# 2. 자체 프레임워크를 정의한다.
# OwnFramework.ipynb 파일에 있는 코드를 여기에 붙여넣는다.
class Linear:
    def __init__(self,nin,nout):
        self.W = np.random.normal(0, 1.0/np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1,nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x):
        self.x=x
        return np.dot(x, self.W.T) + self.b
    
    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        self.dW = dW
        self.db = db
        return dx
    
    def update(self,lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db

#
# N개 입력, N개의 출력
class Softmax:
    #
    # 모든 값에 대해서 exp를 취한 후, exp를 취한 것들의 전체 합으로 나누어준다.
    # exp를 취하는 것은 모든 값이 양수로 변환되기 때문이다.
    # zmax를 빼주는 이유는 exp의 overflow를 방지하기 위함이다.
    def forward(self,z):
        self.z = z
        zmax = z.max(axis=1,keepdims=True)
        expz = np.exp(z-zmax)
        Z = expz.sum(axis=1,keepdims=True)
        return expz / Z
    def backward(self,dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)

#
# N개 입력, N개 출력
class Tanh:
    def forward(self,x):
        y = np.tanh(x)
        self.y = y
        return y
    def backward(self,dy):
        return (1.0-self.y**2)*dy
    
class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y + 1e-9) # 작은 값 추가로 로그(0) 방지
        return -log_prob.mean()
    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / (self.p + 1e-9)

class Net:
    def __init__(self):
        self.layers = []
    
    def add(self,l):
        self.layers.append(l)
        
    def forward(self,x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self,z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z
    
    def update(self,lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)

def get_loss_acc(x,y,loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p,y)
    pred = np.argmax(p,axis=1)
    acc = (pred==y).mean()
    return l,acc

def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), batch_size=32, lr=0.1):
    for i in range(0,len(train_x),batch_size):
        xb = train_x[i:i+batch_size]
        yb = train_labels[i:i+batch_size]

        p = net.forward(xb)
        l = loss.forward(p,yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr)

#
# 3-1. 모델 1
# net = Net()
# net.add(Linear(784,10))
# net.add(Softmax())
# loss = CrossEntropyLoss()

# print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
# train_epoch(net, train_features_norm, train_labels, batch_size=84, lr=0.1)      
# print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, tain_labels)))
# print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_features_norm, test_labels)))

#
# 3-2. 모델 2
#
# 입력, 출력 수는 유지하면서 히든 레이어의 입출력수는 조정하면서 테스트한다.
#
middle, batch_size, lr = 2352, 210, 0.035 # 0.92
net = Net()
net.add(Linear(784,middle))
net.add(Tanh())
net.add(Linear(middle,10))
net.add(Softmax())
loss = CrossEntropyLoss()

print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
train_epoch(net, train_features_norm, train_labels, batch_size=batch_size, lr=lr)      
print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_features_norm, test_labels)))
train_epoch(net, train_features_norm, train_labels, batch_size=batch_size, lr=lr)      
print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_features_norm, test_labels)))

sys.exit(0)

#
# 3-3. 모델 3
middle1, middle2, batch_size, lr = 2352, 4704, 210, 0.035 # 0.92

net = Net()
net.add(Linear(784,middle1))
net.add(Tanh())
net.add(Linear(middle1,middle2))
net.add(Tanh())
net.add(Linear(middle2,10))
net.add(Softmax())
loss = CrossEntropyLoss()

print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
train_epoch(net, train_features_norm, train_labels, batch_size=batch_size, lr=lr)      
print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_features_norm, train_labels)))
print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_features_norm, test_labels)))
