# Copyright (c) Eli Lilly and Company and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

def read_df(df_name):
    """
    Read a dataframe, split into 3 isotypes
    """

    data = pd.read_csv(df_name)
    print("Data shape: ", data.shape)
    mask = data["isotype"].str.startswith("G1")
    data_G1 = data[mask]
    mask = data["isotype"].str.startswith("G4")
    data_G4 = data[mask]
    mask = data["isotype"].str.startswith("G2")
    data_G2 = data[mask]
    print(f"# G1: {data_G1.shape[0]}")
    print(f"# G4: {data_G4.shape[0]}")
    print(f"# G2: {data_G2.shape[0]}")
    if data_G1.shape[0] + data_G4.shape[0] + data_G2.shape[0] != data.shape[0]:
        print("Error: not matching total size")
    return data, data_G1, data_G4, data_G2

def split(Xs, ys, inds, train_size=0.85, random_state=42):
    """
    Split data into train, test
    """
    Xs_train, Xs_test, ys_train, ys_test, inds_train, inds_test = train_test_split(Xs, ys, inds, train_size=train_size, random_state=random_state)
    print(f"Xs_train shape: {Xs_train.shape}")
    print(f"Xs_test shape: {Xs_test.shape}")
    print(f"ys_train shape: {ys_train.shape}")
    print(f"ys_test shape: {ys_test.shape}")
    return Xs_train, Xs_test, ys_train, ys_test, inds_train, inds_test


def kfold_split(path_output, ys_train, num_folds, random_state=42):
    """
    Split train data into (num_folds) folds, and save different folds
    """
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for k, (train_idx, val_idx) in enumerate(kfold.split(ys_train)):
        print("-" * 80)
        print(f"fold {k}, train, val:")
        print(len(train_idx), len(val_idx))
        np.save(os.path.join(f"{path_output}", f"{num_folds}fold_idx_fold_{k}_train.npy"), train_idx)
        np.save(os.path.join(f"{path_output}", f"{num_folds}fold_idx_fold_{k}_val.npy"), val_idx)

def get_kfold(path_output, i, Xs_train, ys_train, num_folds):
    """
    Retrieve train, val data based on Kfolds
    """
    train_idx = np.load(os.path.join(f"{path_output}", f"{num_folds}fold_idx_fold_{i}_train.npy"))
    val_idx = np.load(os.path.join(f"{path_output}", f"{num_folds}fold_idx_fold_{i}_val.npy"))
    Xs_cv_train = Xs_train[train_idx]
    Xs_cv_val = Xs_train[val_idx]
    ys_cv_train = ys_train[train_idx]
    ys_cv_val = ys_train[val_idx]
    return Xs_cv_train, Xs_cv_val, ys_cv_train, ys_cv_val

def load_array(Xs, ys, batch_size, is_train=True):
    dataset = TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(ys).float())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_loader

def extract_seqid(input_fasta):
    seqid_ls = set()
    with open(input_fasta, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):
                line = line[1:].strip()
                if line[-3:] in ["_HC", "_LC"]:
                    line = line[:-3]
                seqid_ls.add(line)
    seqid_ls = sorted(list(seqid_ls))
    return seqid_ls

def organize_embed(path_embed, seqid_ls, EMB_LAYER=33):
    embed_HC_list = []
    embed_LC_list = []

    for seq_id in seqid_ls:
        embed_HC = torch.load(os.path.join(path_embed, str(seq_id) + "_HC.pt"))
        embed_LC = torch.load(os.path.join(path_embed, str(seq_id) + "_LC.pt"))
        mean_representations_HC = embed_HC['mean_representations'][EMB_LAYER]
        mean_representations_LC = embed_LC['mean_representations'][EMB_LAYER]
        embed_HC_list.append(mean_representations_HC)
        embed_LC_list.append(mean_representations_LC)

    embed_HC_list = torch.stack(embed_HC_list, dim=0).numpy()
    embed_LC_list = torch.stack(embed_LC_list, dim=0).numpy()

    # combine HC + LC
    embed_HC_LC_list = np.concatenate((embed_HC_list, embed_LC_list), axis=1)

    #print(embed_HC_LC_list.shape, embed_HC_list.shape, embed_LC_list.shape)
    return embed_HC_LC_list


def extract(path_embed, data, EMB_LAYER=33):
    data_y_H6 = []
    embed_HC_list = []
    embed_LC_list = []
    seq_id_list = []

    for idx, row in data.iterrows():
        seq_id = row.seq_id
        H6 = row.H6_exp_mean
        data_y_H6.append(H6)
        embed_HC = torch.load(os.path.join(path_embed, str(seq_id) + "_HC.pt"))
        embed_LC = torch.load(os.path.join(path_embed, str(seq_id) + "_LC.pt"))
        mean_representations_HC = embed_HC['mean_representations'][EMB_LAYER]
        mean_representations_LC = embed_LC['mean_representations'][EMB_LAYER]
        embed_HC_list.append(mean_representations_HC)
        embed_LC_list.append(mean_representations_LC)
        seq_id_list.append(seq_id)

    embed_HC_list = torch.stack(embed_HC_list, dim=0).numpy()
    embed_LC_list = torch.stack(embed_LC_list, dim=0).numpy()

    # combine HC + LC
    embed_HC_LC_list = np.concatenate((embed_HC_list, embed_LC_list), axis=1)

    print(embed_HC_LC_list.shape, embed_HC_list.shape, embed_LC_list.shape, len(data_y_H6))
    return seq_id_list, embed_HC_LC_list, data_y_H6

def organize_output(predicted_dict, seqid_ls):
    df = pd.DataFrame(predicted_dict, index = seqid_ls)
    df.index.name = "seq_id"
    df.reset_index(level=0, inplace=True)
    for typ in ['H6']:
        try:
            col = df.loc[:, f"model_mlp2_0_{typ}.pt":f"model_mlp2_9_{typ}.pt"]
        except:
            col = df.loc[:, f"model_mlp2_0_{typ}.pt":f"model_mlp2_4_{typ}.pt"]
        df[f'{typ}_pred_mean'] = col.mean(axis=1)
        df[f'{typ}_pred_median'] = col.median(axis=1)
    # only output mean/median 
    subset = ['seq_id', 'H6_pred_mean', 'H6_pred_median']
    df_subset = df[subset]
    print("Final result shape: ", df_subset.shape)
    return df_subset






