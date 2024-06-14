#
# IRIS 데이터셋을 PyTorch Lightning을 사용하여 학습하는 예제
#
import gzip
import pickle
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.datasets import load_iris

iris = load_iris()
data = iris['data']
labels = iris['target']
assert data.shape == (150, 4)
assert labels.shape == (150,)

batch_size = 16

# 데이터를 Tensor로 변환
features = torch.tensor(data, dtype=torch.float32) / 8.0  # 0 ~ 1 범위로 정규화
labels = torch.tensor(labels, dtype=torch.long)

# 데이터셋을 훈련 및 검증용으로 분할
dataset = torch.utils.data.TensorDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 3)
        # self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
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
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.035)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# 모델 학습
model = SimpleNN()
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc', 
    mode='max', 
    save_top_k=1, 
    dirpath='iris_models/', 
    filename='iris-model-{epoch:02d}-{val_acc:.2f}'
)
logger = TensorBoardLogger("iris_logs", name="iris_model")

trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback], logger=logger)
trainer.fit(model, train_loader, test_loader)

# 텐서보드를 통한 시각화
# 터미널에서 실행: tensorboard --logdir=./logs
