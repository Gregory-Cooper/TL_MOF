import sherpa

# help function
from transfer_learning import NeuralNet_sherpa_optimize
from dataset_loader import (
    data_loader,
    all_filter,
    get_descriptors,
    one_filter,
    data_scaler,
)

# modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from Statistics_helper import stratified_cluster_sample
from tqdm import tqdm
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

parameters = [
    sherpa.Continuous(name="lr", range=[0.0002, 0.1], scale="log"),
    # sherpa.Discrete(name='Epoch', range=[10,100]),
    sherpa.Discrete(name="H_l1", range=[10, 300]),
    sherpa.Choice(
        name="activate",
        range=["nn.Hardswish", "nn.PReLU", "nn.ReLU", "nn.Sigmoid", "nn.LeakyReLU"],
    ),
]
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
study = sherpa.Study(
    parameters=parameters,
    algorithm=algorithm,
    lower_is_better=False,
    disable_dashboard=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = os.getcwd()
file_name = "data/CrystGrowthDesign_SI.csv"

"""
Data description.

    Descriptors:
        'void fraction', 'Vol. S.A.', 'Grav. S.A.', 'Pore diameter Limiting', 'Pore diameter Largest'
    Source task:
        'H2@100 bar/243K (wt%)'
    Target tasks:
        'H2@100 bar/130K (wt%)' 'CH4@100 bar/298 K (mg/g)' '5 bar Xe mol/kg' '5 bar Kr mol/kg'
"""

descriptor_columns = [
    "void fraction",
    "Vol. S.A.",
    "Grav. S.A.",
    "Pore diameter Limiting",
    "Pore diameter Largest",
]
one_filter_columns = ["H2@100 bar/243K (wt%)"]
another_filter_columns = ["H2@100 bar/130K (wt%)"]

# load data
data = data_loader(base_path, file_name)

# extract descriptors and gas adsorptions
one_property = one_filter(data, one_filter_columns)
descriptors = get_descriptors(data, descriptor_columns)

# prepare training inputs and outputs
X = np.array(descriptors.values, dtype=np.float32)
y = np.array(one_property.values, dtype=np.float32).reshape(len(X),)
X = data_scaler(X)
y = data_scaler(y.reshape(-1, 1)).reshape(len(X),)

# makes transfer trials... more of a legacy code ---- function cannot be pulled out of .py bc of data dependencies
def transfer_learning(
    s_param, learning_rate, transfer=False, nsamples=None, nbatches=None, names=None
):
    seeds = np.arange(nbatches)
    Ns = list()
    scores_epochs = list()
    scores_test = list()
    scores_train = list()

    pred_tests = list()
    grt_train_X = list()
    grt_test_X = list()
    grt_tests = list()

    for seed in seeds:
        X_train,X_test,y_train,y_test=stratified_cluster_sample(data,descriptor_columns,one_filter_columns[0],5)

        ## model, loss, and optimizer
        if transfer:
            model = NeuralNet_sherpa_optimize(5,1,s_param).to(device)
            model.load_state_dict(torch.load("temp_model.ckpt"))
            model.fc1.weight.requires_grad = False
            model.fc1.bias.requires_grad = False
            model.fc2.weight.requires_grad = False
            model.fc2.bias.requires_grad = False

            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
        else:
            model = NeuralNet_sherpa_optimize(5,1,s_param).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        ## train, val, test data split
        X_train, X_test, y_train, y_test = train_test_split(
            X_small, y_small, test_size=0.1, random_state=1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=1
        )

        scores_epoch = list()
        num_epochs = 5000
        N = 0
        for epoch in range(num_epochs):
            inputs = torch.from_numpy(X_train)
            labels = torch.from_numpy(y_train)

            outputs = model(inputs).view(-1,)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            N += 1
            if epoch % 5 == 0:
                inputs_val = torch.from_numpy(X_val)
                labels_val = torch.from_numpy(y_val)

                outputs_val = model(inputs_val).view(-1,)
                score = r2_score(labels_val.data.numpy(), outputs_val.data.numpy())
                #         print('Predictive accuracy on validation set at epoch {}/{} is {}'.format(epoch, num_epochs, score))
                scores_epoch.append(score)
            if len(scores_epoch) >= 2:
                if score < scores_epoch[-2]:
                    break
        scores_epochs.append(scores_epoch)
        Ns.append(N)

        score_train = r2_score(
            torch.from_numpy(y_train).data.numpy(),
            model(torch.from_numpy(X_train)).view(-1,).data.numpy(),
        )
        #         score_train = mean_squared_error(torch.from_numpy(y_train).data.numpy(), model(torch.from_numpy(X_train)).view(-1,).data.numpy())
        scores_train.append(score_train)

        pred_tests.append(model(torch.from_numpy(X_test)).view(-1,).data.numpy())
        grt_train_X.append(torch.from_numpy(X_train).data.numpy())
        grt_test_X.append(torch.from_numpy(X_test).data.numpy())
        grt_tests.append(torch.from_numpy(y_test).data.numpy())
        score_test = r2_score(
            torch.from_numpy(y_test).data.numpy(),
            model(torch.from_numpy(X_test)).view(-1,).data.numpy(),
        )
        #         score_test = mean_squared_error(torch.from_numpy(y_test).data.numpy(), model(torch.from_numpy(X_test)).view(-1,).data.numpy())
        scores_test.append(score_test)
        torch.save(model, f"{name}.pt")

    return scores_train, scores_test, grt_train_X, grt_test_X


for trial in study:
    learning_rate = trial.parameters["lr"]
    # batch=trial.parameters["Epoch"]
    ## model, loss, and optimizer
    # always used 5 features to make 1 prediction hence 5,1
    model = NeuralNet_sherpa_optimize(5, 1, trial.parameters).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ## train, val, test data split
    # note these are not split by cluster yet.....
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    # Training
    scores_epochs = list()
    num_epochs = 5000

    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train)

        outputs = model(inputs).view(-1,)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            inputs_val = torch.from_numpy(X_val)
            labels_val = torch.from_numpy(y_val)

            outputs_val = model(inputs_val).view(-1,)
            score = r2_score(labels_val.data.numpy(), outputs_val.data.numpy())
            score_train = r2_score(
                torch.from_numpy(y_train).data.numpy(),
                model(torch.from_numpy(X_train)).view(-1,).data.numpy(),
            )
            study.add_observation(
                trial=trial,
                iteration=epoch,
                objective=score,
                context={"training_error": score_train},
            )

    #         print('Predictive accuracy on validation set at epoch {}/{} is {}'.format(epoch, num_epochs, score))
    score_test = r2_score(
        torch.from_numpy(y_test).data.numpy(),
        model(torch.from_numpy(X_test)).view(-1,).data.numpy(),
    )
    score_train = r2_score(
        torch.from_numpy(y_train).data.numpy(),
        model(torch.from_numpy(X_train)).view(-1,).data.numpy(),
    )
    torch.save(model.state_dict(), "temp_model.ckpt")
    study.add_observation(
        trial=trial, objective=score_test, context={"training_error": score_train}
    )
