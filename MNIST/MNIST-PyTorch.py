#
# MNIST 문제를 PyTorch로 풀어보자.
#
import gzip
import pickle
import torch
import matplotlib.pyplot as plt

#
# 1. 데이터를 준비한다.
with gzip.open('../lessons/data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

data = MNIST['Train']['Features']
labels = MNIST['Train']['Labels']
assert data.shape == (42000, 784)
assert labels.shape == (42000,)


# 데이터를 Tensor로 변환
features = torch.tensor(data, dtype=torch.float32) / 255.0  # 0 ~ 1 범위로 정규화
labels = torch.tensor(labels, dtype=torch.long)

# 데이터셋을 훈련 및 검증용으로 분할
dataset = torch.utils.data.TensorDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=210, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=210, shuffle=False)

# 모델 정의
class SimpleNN(torch.nn.Module):
    def __init__(self, hidden_size=2352):
        super(SimpleNN, self).__init__()
        # 784 -> 2352 -> 10
        self.fc1 = torch.nn.Linear(28*28, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = SimpleNN()

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.035)

# 모델 학습
num_epochs = 10
train_acc_list, val_acc_list = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_acc = correct_train / total_train
    train_acc_list.append(train_acc)
    
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    
    val_acc = correct_test / total_test
    val_acc_list.append(val_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
          f'Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

# 학습 결과 시각화
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.legend()
plt.show()