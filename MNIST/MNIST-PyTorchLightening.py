#
# MNIST 데이터셋을 PyTorch Lightning을 사용하여 학습하는 예제
# chatGPT-4 활용
#
import gzip
import pickle

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# MNIST 데이터셋 로드
with gzip.open('../lessons/data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle, encoding='latin1')

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=210, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=210, shuffle=False, num_workers=0)

class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 2352)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(2352, 10)
        # self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.035)
        return optimizer

# 모델 학습
model = SimpleNN()
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc', 
    mode='max', 
    save_top_k=1, 
    dirpath='models/', 
    filename='mnist-model-{epoch:02d}-{val_acc:.2f}'
)
logger = TensorBoardLogger("logs", name="mnist_model")

trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback], logger=logger)
trainer.fit(model, train_loader, test_loader)

# 텐서보드를 통한 시각화
# 터미널에서 실행: tensorboard --logdir=./logs
