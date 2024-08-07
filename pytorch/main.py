import torch
from torch import nn
import numpy as np

torch.manual_seed(0)
device = "cuda"
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.max_memory_allocated())

## DATASET

def create_data(samples, classes):
    np.random.seed(0)
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)


BATCH_LEN = 1000
INTERATIONS = 10000
N_CLASSES = 2

train_data, target = create_data(BATCH_LEN, N_CLASSES)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 2),
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
for i in range(0, INTERATIONS):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = loss_fn(outputs, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % BATCH_LEN == BATCH_LEN-1:
        last_loss = running_loss / BATCH_LEN # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        running_loss = 0.