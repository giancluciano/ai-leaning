import torch
import pandas as pd
from torch import nn
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
device = "cuda"

train = pd.read_csv('data/titanic/train.csv')
FEATURE_COLUMNS = ['Pclass', 'Sex', 'Fare']
TARGET_COLUMN = ['Survived']
EXTRA_FEATURE_COLUMNS = ['Ticket', 'Age'] + FEATURE_COLUMNS

## INPUTS
train['Sex'] = le.fit_transform(train['Sex'])
train['Ticket'] = le.fit_transform(train['Ticket'])
train['Age'] = train['Age'].fillna(0)
train.describe(include='all')
train[EXTRA_FEATURE_COLUMNS + TARGET_COLUMN].corr()[TARGET_COLUMN]

input = train[EXTRA_FEATURE_COLUMNS]
input = torch.from_numpy(input.to_numpy()).to(device)


## TARGET
target = torch.from_numpy(train[TARGET_COLUMN].to_numpy().flatten()).to(device)

EPOCHS = 200
BATCH_LEN = len(input)
INTERATIONS = BATCH_LEN * EPOCHS
N_CLASSES = 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.double()

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    

torch.manual_seed(0)
model = NeuralNetwork().to("cuda")


loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.
last_loss = 0
last_accuracy = 0
for i in range(0, INTERATIONS):
    optimizer.zero_grad()
    outputs = model(input)
    loss = loss_fn(outputs, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % BATCH_LEN == BATCH_LEN-1:
        last_accuracy = torch.sum(outputs.max(1).indices == target)/ BATCH_LEN
        last_loss = running_loss / BATCH_LEN # loss per batch
        print('  batch {} loss: {} accuracy {:%}'.format(i + 1, last_loss, last_accuracy))
        running_loss = 0.