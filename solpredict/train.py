# Copyright (c) Eli Lilly and Company and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



# Given a new dataframe, update the model using K fold CV + hyper-parameter selection
"""Inputs: config.yaml
1. dataframe: seq_id, isotype, H6_exp_mean
2. path_embed (optional): path storing the embeddings of all the molecules
3. hyper-parameter list + k (CV fold) + typ (which model to train: H6) + is_split (whether split into train/test)
4. path_output: which folder to store the outputs

Outputs:
trained_models_update_cv: results for different combinations
for each combo:
    fold_*: model (*.pt) and results for each fold
    params.txt: params used
cv_summary.txt: Best parameter info and Best metric results
"""
import sys
import yaml
from pathlib import Path
from data import organize_embed, extract, read_df, kfold_split, get_kfold, load_array, organize_output
from evaluate import eval_model
from model import MLP2Layer, init_weights
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from collections import defaultdict
import itertools
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# set seed, same result
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 42
set_seed(seed)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """
    train a model 1 time
    """
    #since = time.time()

    loss_plot_train, loss_plot_val = [], []
    spearmanr_plot_train, spearmanr_plot_val = [], []
    r2_plot_train, r2_plot_val = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_r2 = 0.0

    # start training
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    preds = model(inputs)
                    # loss
                    loss = criterion(preds, labels)
                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            dataset = dataloaders[phase].dataset
            Xs, ys = [], []
            for inputs, labels in dataset:
                Xs.append(inputs.detach().numpy())
                ys.append(labels.detach().numpy())

            Xs = np.array(Xs)
            ys = np.array(ys)

            epoch_loss = running_loss / len(dataset)

            with torch.set_grad_enabled(False):
                predicted = model(torch.from_numpy(Xs)).detach().numpy()
                spearmanr = scipy.stats.spearmanr(ys, predicted)[0]
                r2 = r2_score(ys, predicted)

            if phase == 'train':
                loss_plot_train.append(epoch_loss)
                spearmanr_plot_train.append(spearmanr)
                r2_plot_train.append(r2)
                if scheduler:
                    scheduler.step()
            else:
                loss_plot_val.append(epoch_loss)
                spearmanr_plot_val.append(spearmanr)
                r2_plot_val.append(r2)

            if phase == 'val' and r2 > best_r2:
                best_r2 = r2
                best_model_wts = copy.deepcopy(model.state_dict())

    # print()
    #time_elapsed = time.time() - since

    # loading best model weights
    model.load_state_dict(best_model_wts)

    performance = defaultdict(lambda: defaultdict(list))
    performance["loss_plot_train"] = loss_plot_train
    performance["loss_plot_val"] = loss_plot_val
    performance["spearmanr_plot_train"] = spearmanr_plot_train
    performance["spearmanr_plot_val"] = spearmanr_plot_val
    performance["r2_plot_train"] = r2_plot_train
    performance["r2_plot_val"] = r2_plot_val

    return model, best_r2, performance


def train_kfold(path_model, num_folds, Xs_train, ys_train, curr_params):
    """
    Given a set of params, train a model k fold
    return: models, best_r2 list, avg r2
    """
    models = []
    r2_ls = []
    performance_ls = []

    n_input, n_output = curr_params['n_input'], curr_params['n_output']
    batch_size = curr_params['batch_size']
    n_hidden_1 = curr_params['n_hidden_1']
    n_hidden_2 = curr_params['n_hidden_2']
    lr = curr_params['lr']
    num_epochs = curr_params['num_epochs']

    for i in range(num_folds):
        Xs_cv_train, Xs_cv_val, ys_cv_train, ys_cv_val = get_kfold(path_model, i, Xs_train, ys_train, num_folds)
        train_loader = load_array(Xs_cv_train, ys_cv_train, batch_size, is_train=True)
        val_loader = load_array(Xs_cv_val, ys_cv_val, batch_size, is_train=False)
        dataloaders = {'train': train_loader, 'val': val_loader}

        # initiate the model
        model = MLP2Layer(n_input, n_hidden_1, n_hidden_2, n_output)
        model.apply(init_weights)

        # define loss func and optimizer
        criterion = nn.MSELoss()  # mean squared loss for regression
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None

        # train
        model, best_r2, performance = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs)

        # save the best model based on R2
        models.append(model)
        r2_ls.append(best_r2)
        performance_ls.append(performance)

    avg_r2 = np.mean(r2_ls)
    print(f"avg_r2: {avg_r2}")
    return models, r2_ls, avg_r2, performance_ls


def cross_validate(typ, path_model, Xs_train, ys_train, params, num_folds=5):
    # Compute all param combs
    param_sets = defaultdict(dict)
    sorted_keys = sorted(params.keys())
    param_combos = itertools.product(*(params[s] for s in sorted_keys))
    counter = 0
    for p in list(param_combos):
        curr_params = list(p)
        param_dict = dict(zip(sorted_keys, curr_params))
        if param_dict['n_hidden_1'] > param_dict['n_hidden_2']:
            param_sets[counter] = param_dict
            counter += 1
    print("Total param combs: ", len(param_sets.keys()))

    best_avg_r2 = 0.0
    best_avg_r2_ls = []
    best_combo = None
    best_params = dict()  # best param combo
    best_models = []  # best five models with the highest avg R2
    best_performance_ls = []
    avg_r2_ls = []

    # For each combo
    for counter in param_sets.keys():
        curr_params = param_sets[counter]
        print("-" * 80)
        print('{}\t{}'.format(counter, curr_params))

        # k fold CV
        models, r2_ls, avg_r2, performance_ls = train_kfold(path_model, num_folds, Xs_train, ys_train, curr_params)

        # saving
        print("Saving results...")

        for i in range(num_folds):
            path_model_cur = f"{path_model}/combo_{counter}/fold_{i}/"
            title = f"combo_{counter}_fold_{i}"
            Path(path_model_cur).mkdir(parents=True, exist_ok=True)
            model = models[i]
            r2 = r2_ls[i]
            performance = performance_ls[i]

            # save the model checkpoint
            torch.save(model.state_dict(), f'{path_model}/combo_{counter}/model_mlp2_{i}_{typ}.pt')
            Xs_cv_train, Xs_cv_val, ys_cv_train, ys_cv_val = get_kfold(path_model, i, Xs_train, ys_train, num_folds)
            eval_model(title, path_model_cur, model, Xs_cv_train, Xs_cv_val, ys_cv_train, ys_cv_val, performance)

        with open(f"{path_model}/combo_{counter}/params.txt", 'w') as f:
            f.write('{}\t{}'.format(counter, curr_params))

        avg_r2_ls.append(avg_r2)
        if avg_r2 > best_avg_r2:
            best_combo = counter
            best_avg_r2 = avg_r2
            best_avg_r2_ls = r2_ls
            best_params = curr_params
            best_models = models
            best_performance_ls = performance_ls

    with open(f"{path_model}/cv_summary.txt", 'w') as f:
        f.write('Best params: combo {}\n{}'.format(best_combo, best_params))
        f.write('\n')
        f.write('Best avg r2: \n{}'.format(best_avg_r2))
    np.save(f"{path_model}/avg_r2_ls_{len(avg_r2_ls)}.npy", avg_r2_ls)
    return best_models, best_params, best_avg_r2_ls, best_avg_r2, best_performance_ls

def main(config):
    # Read yaml file to get keywords
    with open(config) as f:
        read_data = yaml.load(f, Loader=yaml.SafeLoader)
    print("Reading args: ")
    print(read_data)
    path_data = read_data['path_data']
    input_fasta = read_data['input_fasta']
    df_name = read_data['df_name']
    path_embed = read_data['path_embed']
    path_output = read_data['path_output']
    typ = read_data['typ']
    is_split = read_data['is_split']
    num_folds = read_data['num_folds']
    params = read_data['params']
    path_output = os.path.join(path_output, typ)
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Extract esm1b embeddings
    if os.path.exists(path_embed):
        print('embed exists, skip embed computation')
    else:
        print('compute embed ' + '-'*10)
        output_dir_embed = os.path.join(path_output, 'embed')
        Path(output_dir_embed).mkdir(parents=True, exist_ok=True)
        cmd = f"python extract.py esm1b_t33_650M_UR50S {input_fasta} {output_dir_embed} --repr_layers 33 --include mean per_tok"
        os.system(cmd)
        path_embed = output_dir_embed

    # Read data
    data, data_G1, data_G4, data_G2 = read_df(os.path.join(path_data, df_name))
    seq_id_list, embed_HC_LC_list, data_y_H6 = extract(path_embed=path_embed, data=data)
    if typ == 'H6':
        Xs = embed_HC_LC_list
        ys = data_y_H6
        ys = np.asarray(ys).reshape(len(ys), 1)
    else:
        print('Error, unrecognized training type')
    # Split into Kfold
    kfold_split(path_output, ys, num_folds)
    # Cross validation
    best_models, best_params, best_avg_r2_ls, best_avg_r2, best_performance_ls = cross_validate(typ, path_output, Xs, ys, params, num_folds)
    print("Finish training")
    print(f"Best avg R2: {best_avg_r2}")

if __name__ == "__main__":
    main(sys.argv[1])




