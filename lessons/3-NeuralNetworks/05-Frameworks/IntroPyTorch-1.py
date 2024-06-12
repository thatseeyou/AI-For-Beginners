#
# IntroPyTorch.ipynb의 Example 2: Classification 부분을 Python 파일로 변환한 것이다.
#
import torch
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

#
# 0. 데이터를 시각화하는 함수를 정의한다.
def plot_dataset(features, labels, W=None, b=None):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    if W is not None:
        min_x = min(features[:,0])
        max_x = max(features[:,0])
        min_y = min(features[:,1])*(1-.1)
        max_y = max(features[:,1])*(1+.1)
        cx = np.array([min_x,max_x],dtype=np.float32)
        # W[0]*x + W[1]*y + b - 0.5 = 0
        # 0.5는 중간값을 의미한다.
        cy = (0.5-W[0]*cx-b)/W[1]
        ax.plot(cx,cy,'g')
        ax.set_ylim(min_y,max_y)
    # fig.show()
    # ipynb가 아닌 경우에는 아래 코드를 사용한다.
    plt.show()
    # plt.draw()
    # plt.pause(3)  # 0.001초 동안 그래프를 표시하고 코드 실행을 계속

#
# 1. 데이터를 준비한다.
# 1-1. 데이터를 생성한다.
np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
# 정규 분포를 갖는 두 개의 특성(0, 1)을 갖는 데이터를 생성한다.
# flip_y=0.1은 10%의 노이즈를 추가한다.
# class_sep은 두 클래스의 분리 정도를 결정한다.
# n_informative=2는 두 개의 특성이 클래스를 결정하는 데 중요하다는 것을 의미한다.
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.1,class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

#
# 1-2. 70%의 데이터를 훈련 데이터로, 15%의 데이터를 검증 데이터로, 나머지 15%를 테스트 데이터로 사용한다.
split = [ 70*n//100, (15+70)*n//100 ]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)

#
# 1-3. 데이터를 시각화한다.
# plot_dataset(train_x, train_labels)

#
# 1-4. 데이터를 PyTorch의 Tensor로 변환한다.
dataset = torch.utils.data.TensorDataset(torch.tensor(train_x),torch.tensor(train_labels,dtype=torch.float32))
# DataLoader의 역할은 데이터를 미니배치로 나누어주는 것이다.
dataloader = torch.utils.data.DataLoader(dataset,batch_size=16)

#
# 2. 신경망을 정의한다.
class Network():
  def __init__(self):
     self.W = torch.randn(size=(2,1),requires_grad=True)
     self.b = torch.zeros(size=(1,),requires_grad=True)

  def forward(self,x):
    return torch.matmul(x,self.W)+self.b

  def zero_grad_original(self):
    self.W.data.zero_()
    self.b.data.zero_()

  # chatGPT가 수정 권고한 코드
  def zero_grad(self):
    if self.W.grad is not None:
      self.W.grad.zero_()
    if self.b.grad is not None:
      self.b.grad.zero_()

  def update(self,lr=0.1):
    self.W.data.sub_(lr*self.W.grad)
    self.b.data.sub_(lr*self.b)

net = Network()

#
# 3. 신경망을 훈련한다.
# 3-1. 한 번의 미니배치에 대한 훈련을 수행하는 함수를 정의한다.
def train_on_batch(net, x, y):
  z = net.forward(x).flatten()
  loss = torch.nn.functional.binary_cross_entropy_with_logits(input=z,target=y)
  net.zero_grad()
  # net.W.grad, net.b.grad를 계산한다.
  loss.backward()
  # net.W.data, net.b.data를 업데이트한다.
  net.update()
  return loss

#
# 3-2. 15번의 에포크 동안 훈련을 수행한다.
for epoch in range(15):
  for (x, y) in dataloader:
    loss = train_on_batch(net,x,y)
  print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

print(net.W,net.b)

plot_dataset(train_x, train_labels, net.W.detach().numpy(), net.b.detach().numpy())