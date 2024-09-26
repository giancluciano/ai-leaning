from numpy import random
from ray import tune
from ray import train
from torch import nn, manual_seed, no_grad, max, from_numpy, optim
import pandas as pd
from sklearn import preprocessing
from ray.tune.search.hyperopt import HyperOptSearch
from ray import init
from math import floor
from matplotlib import pyplot
from hyperopt import hp

le = preprocessing.LabelEncoder()

device = "cuda"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 64),
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
    
EPOCH_SIZE = 10000


def train_func(model, optimizer, loss_fn, input, target):
    
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()


def test_func(model, input, target):
    correct = 0
    total = 0
    with no_grad():
        outputs = model(input)
        _, predicted = max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return correct / total

def train_mnist(config):
    manual_seed(0)
    data = pd.read_csv('/home/gian/Projects/ai-leaning/pytorch/data/titanic/train.csv')
    FEATURE_COLUMNS = ['Pclass', 'Sex', 'Fare']
    TARGET_COLUMN = ['Survived']
    data['Sex'] = le.fit_transform(data['Sex'])
    men_data = data.loc[data['Sex'] == 1]
    women_data = data.loc[data['Sex'] == 0]
    men_input = men_data[FEATURE_COLUMNS]
    men_input = from_numpy(men_input.to_numpy()).to(device)
    men_target = from_numpy(men_data[TARGET_COLUMN].to_numpy().flatten()).to(device)

    men_model = NeuralNetwork().to(device)
    women_model = NeuralNetwork().to(device)

    optimizer = optim.SGD(men_model.parameters(), lr=config["lr"], momentum=config["momentum"])
    loss_fn = nn.CrossEntropyLoss()
    for i in range(EPOCH_SIZE):
        train_func(men_model, optimizer, loss_fn, men_input, men_target)
        if i % 1000 == 0:
            result = test_func(men_model, men_input, men_target)
            print(f"men mean_accuracy {result:.2%}")
        #train.report({"men mean_accuracy": result})

    women_input = women_data[FEATURE_COLUMNS]
    women_input = from_numpy(women_input.to_numpy()).to(device)
    women_target = from_numpy(women_data[TARGET_COLUMN].to_numpy().flatten()).to(device)
    optimizer = optim.SGD(women_model.parameters(), lr=config["lr"], momentum=config["momentum"])
    loss_fn = nn.CrossEntropyLoss()
    for i in range(EPOCH_SIZE):
        train_func(women_model, optimizer, loss_fn, women_input, women_target)
        if i % 1000 == 0:
            result = test_func(women_model, women_input, women_target)
            print(f'women mean_accuracy {result:.2%}')
        #train.report({"women mean_accuracy": result})

    test_data = pd.read_csv('/home/gian/Projects/ai-leaning/pytorch/data/titanic/test.csv')
    test_data['Sex'] = le.fit_transform(test_data['Sex'])
    men_data = test_data.loc[test_data['Sex'] == 1]
    women_data = test_data.loc[test_data['Sex'] == 0]
    
    men_input = men_data[FEATURE_COLUMNS]
    men_input = from_numpy(men_input.to_numpy()).to(device)
    men_outputs = men_model(men_input)
    _, men_predicted = max(men_outputs.data, 1)

    women_input = women_data[FEATURE_COLUMNS]
    women_input = from_numpy(women_input.to_numpy()).to(device)
    women_outputs = women_model(women_input)
    _, women_predicted = max(women_outputs.data, 1)


    test_data.loc[test_data['Sex'] == 1, 'Survived'] = men_predicted.tolist()
    test_data.loc[test_data['Sex'] == 0, 'Survived'] = women_predicted.tolist()
    test_data['Survived'] = test_data['Survived'].astype('Int64')
    test_data[['PassengerId', 'Survived']].to_csv('/home/gian/Projects/ai-leaning/pytorch/data/titanic/result.csv', index=False)


if __name__ == "__main__":

    search_space = {
        "lr": 0.001,
        "momentum": 0.9,
    }
    train_mnist(search_space)