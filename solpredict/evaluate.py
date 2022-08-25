# Copyright (c) Eli Lilly and Company and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Compute various metrics
import os
import scipy
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

dpi=300
sizeOfFont=20
linewidth=2
figsize = (6, 6)

font = {'family':'Arial',
        'weight': 'normal',
        'size': sizeOfFont,
        }

mpl.rc('font', **font)

ticks_font = font_manager.FontProperties(family='Arial', style='normal',
    size=sizeOfFont, weight='normal', stretch='normal')

ticks_font_small = font_manager.FontProperties(family='Arial', style='normal',
    size=16, weight='normal', stretch='normal')

fontdict={'family':'Arial', 'size':'14'}
fontdict_small={'family':'Arial', 'size':'12'}

def save_fig(path_image, fig_id, tight_layout=True, fig_extension="png", resolution=dpi):
    path = os.path.join(path_image, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, transparent=True, bbox_inches ="tight")


def train_val_plot(title, path_model, loss_plot_train, loss_plot_val, spearmanr_plot_train, spearmanr_plot_val,
                   r2_plot_train, r2_plot_val):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    # Loss plot
    ax[0].plot(loss_plot_train, label="Training Loss")
    ax[0].plot(loss_plot_val, label="Validation Loss")
    ax[0].legend(fontsize=18)
    ax[0].grid(True)
    ax[0].set_title("Training loss", fontsize=20);
    ax[0].set_xlabel("Epoch", fontsize=18);
    ax[0].set_ylabel("Loss", fontsize=18);
    # Performance plot
    ax[1].plot(spearmanr_plot_train, label="Training Spearmanr")
    ax[1].plot(spearmanr_plot_val, label="Validation Spearmanr")
    ax[1].plot(r2_plot_train, label="Training R2")
    ax[1].plot(r2_plot_val, label="Validation R2")
    ax[1].legend(fontsize=18)
    ax[1].grid(True)
    ax[1].set_title("Training correlation", fontsize=20);
    ax[1].set_xlabel("Epoch", fontsize=18);
    ax[1].set_ylabel("Spearmanr/R2", fontsize=18);
    save_fig(path_model, f"Training_plot_{title}")

def eval_model(title, path_model, model, Xs_train, Xs_val, ys_train, ys_val, performance):
    model.eval()
    predicted_train = model(torch.from_numpy(Xs_train)).detach().numpy()
    predicted_val = model(torch.from_numpy(Xs_val)).detach().numpy()
    loss_plot_train = performance["loss_plot_train"]
    loss_plot_val = performance["loss_plot_val"]
    spearmanr_plot_train = performance["spearmanr_plot_train"]
    spearmanr_plot_val = performance["spearmanr_plot_val"]
    r2_plot_train = performance["r2_plot_train"]
    r2_plot_val = performance["r2_plot_val"]

    train_val_plot(title, path_model, loss_plot_train, loss_plot_val, spearmanr_plot_train, spearmanr_plot_val,
                   r2_plot_train, r2_plot_val)

    # saving
    np.save(path_model + f"ys_train_true_{len(ys_train)}.npy", ys_train)
    np.save(path_model + f"ys_val_true_{len(ys_val)}.npy", ys_val)
    np.save(path_model + f"ys_train_pred_{len(predicted_train)}.npy", predicted_train)
    np.save(path_model + f"ys_val_pred_{len(predicted_val)}.npy", predicted_val)

    np.save(path_model + f"loss_plot_train_{len(predicted_train)}.npy", loss_plot_train)
    np.save(path_model + f"spearmanr_plot_train_{len(predicted_train)}.npy", spearmanr_plot_train)
    np.save(path_model + f"r2_plot_train_{len(predicted_train)}.npy", r2_plot_train)

    np.save(path_model + f"loss_plot_val_{len(predicted_val)}.npy", loss_plot_val)
    np.save(path_model + f"spearmanr_plot_val_{len(predicted_val)}.npy", spearmanr_plot_val)
    np.save(path_model + f"r2_plot_val_{len(predicted_val)}.npy", r2_plot_val)

    with open(path_model + f"results_summary.txt", 'w') as f:
        f.write(f'Train spearmanr: {scipy.stats.spearmanr(ys_train, predicted_train)}\n')
        f.write(f'Val spearmanr: {scipy.stats.spearmanr(ys_val, predicted_val)}\n')
        pr = scipy.stats.pearsonr(ys_train[:, 0], predicted_train[:, 0])
        f.write(f'Train pearsonr: {pr}\n')
        pr = scipy.stats.pearsonr(ys_val[:, 0], predicted_val[:, 0])
        f.write(f'Val pearsonr: {pr}\n')

        f.write(f'Train r2: {r2_score(ys_train, predicted_train)}\n')
        f.write(f'Val r2: {r2_score(ys_val, predicted_val)}\n')

        f.write(f'Train RMSE: {sqrt(mean_squared_error(ys_train, predicted_train))}\n')
        f.write(f'Val RMSE: {sqrt(mean_squared_error(ys_val, predicted_val))}\n')

    x_min = min(np.min(ys_train), np.min(ys_val))
    x_max = max(np.max(ys_train), np.max(ys_val))

    x = np.linspace(x_min, x_max, 100)

    fig_dims = (14, 6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_dims, sharex=False, sharey=True)
    ax1.scatter(ys_train, predicted_train, c="green")
    ax1.plot(x, x, "--", color="black", label=r'y = x', linewidth=linewidth)

    ax2.scatter(ys_val, predicted_val, c="orange")
    ax2.plot(x, x, "--", color="black", label=r'y = x', linewidth=linewidth)

    ax1.set_xlabel('True')
    ax2.set_xlabel('True')
    ax1.set_ylabel('Pred')
    ax1.set_title('Train')
    ax2.set_title('Val')
    save_fig(path_model, f"Eval_{len(ys_val)}_{title}")

