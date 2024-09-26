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
    
EPOCH_SIZE = 100000

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
    EXTRA_FEATURE_COLUMNS = ['Ticket', 'Age'] + FEATURE_COLUMNS
    TARGET_COLUMN = ['Survived']
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Ticket'] = le.fit_transform(data['Ticket'])
    data['Age'] = data['Age'].fillna(0)
    input = data[FEATURE_COLUMNS]
    input = from_numpy(input.to_numpy()).to(device)
    target = from_numpy(data[TARGET_COLUMN].to_numpy().flatten()).to(device)
    model = NeuralNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    loss_fn = nn.CrossEntropyLoss()
    for i in range(EPOCH_SIZE):
        train_func(model, optimizer, loss_fn, input, target)
        if i % 1000 == 0:
            result = test_func(model, input, target)
            print(f'{result:%}')
        train.report({"mean_accuracy": result})
    
    test_data = pd.read_csv('/home/gian/Projects/ai-leaning/pytorch/data/titanic/test.csv')
    test_data['Sex'] = le.fit_transform(test_data['Sex'])
    test_data['Ticket'] = le.fit_transform(test_data['Ticket'])
    test_data['Age'] = test_data['Age'].fillna(0)
    input = test_data[FEATURE_COLUMNS]

    input = from_numpy(input.to_numpy()).to(device)
    outputs = model(input)
    _, predicted = max(outputs.data, 1)
    test_data['Survived'] = predicted.tolist()
    test_data[['PassengerId', 'Survived']].to_csv('/home/gian/Projects/ai-leaning/pytorch/data/titanic/result.csv', index=False)


if __name__ == "__main__":

    if False: # model tuning
        init(num_gpus=1, num_cpus=12)
        search_space = {
            "lr": hp.loguniform("lr", -10, -1),
            "momentum": hp.uniform("momentum", 0.1, 0.9),
        }

        hyperopt_search = HyperOptSearch(search_space, metric="mean_accuracy", mode="max")

        tuner = tune.Tuner(
            tune.with_resources(
                train_mnist,
                resources={"gpu": 0.8, "cpu": 10}
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=10,
                search_alg=hyperopt_search,
            )
        )

        #results = tuner.fit()
        #print(results)
        #dfs = {result.path: result.metrics_dataframe for result in results}
        #[d.mean_accuracy.plot() for d in dfs.values()]
        #pyplot.show()


    search_space = {
        "lr": 0.001,
        "momentum": 0.9,
    }
    train_mnist(search_space)