import torch

import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from models.neural_networks import *


def accuracy(x: np.ndarray, y: np.ndarray):
    # calculate accuracy for each class and macro F1-score.
    # designed for this task only.
    # return correct number of samples corresponds to each class and the number of samples for each class.
    # for the overall mean accuracy, just sum them all and divide.
    index_0 = np.where(y == 0)[0]
    index_1 = np.where(y == 1)[0]
    index_2 = np.where(y == 2)[0]
    x_0, y_0 = x[index_0], y[index_0]
    x_1, y_1 = x[index_1], y[index_1]
    x_2, y_2 = x[index_2], y[index_2]
    acc_0 = 0 if not len(y_0) or not len(x_0) else accuracy_score(y_0, x_0[..., 0], normalize=False)
    acc_1 = 0 if not len(y_1) or not len(x_1) else accuracy_score(y_1, x_1[..., 0], normalize=False)
    acc_2 = 0 if not len(y_2) or not len(x_2) else accuracy_score(y_2, x_2[..., 0], normalize=False)
    return [acc_0, acc_1, acc_2], [len(index_0), len(index_1), len(index_2)]


def eval_net(model, loader, device):
    model.eval()
    n_val = len(loader)  # the number of batch
    total_acc_0 = 0
    len_0 = 0
    total_acc_1 = 0
    len_1 = 0
    total_acc_2 = 0
    len_2 = 0

    with tqdm(total=n_val, desc='Evaluation round', unit='batch', leave=False) as pbar:
        for bt in loader:
            xs, ys = bt['feature'], bt['label']
            if isinstance(model, GRU):
                xs = xs.long()
            xs = xs.to(device=device)
            ys = ys.to(device=device)

            with torch.no_grad():
                if isinstance(model, ANN) or isinstance(model, GRU):
                    xs = torch.squeeze(xs)
                refs = model(xs)
                refs = F.softmax(refs, dim=1)
                refs = torch.argmax(refs, dim=1, keepdim=True)
            accuracies, lens = accuracy(refs.detach().cpu().numpy(), ys.detach().cpu().numpy())
            total_acc_0 += accuracies[0]
            len_0 += lens[0]
            total_acc_1 += accuracies[1]
            len_1 += lens[1]
            total_acc_2 += accuracies[2]
            len_2 += lens[2]
            pbar.update()

    model.train()
    return total_acc_0 / len_0, total_acc_1 / len_1, total_acc_2 / len_2


def evaluation_metrics(x: np.ndarray, y: np.ndarray):
    # return overall accuracy, accuracy of each class, macro F1_score based on input variables.
    # please note that is function should be performed on the whole dataset level.
    # x and y should has a consistent shape.
    x_0, y_0 = x[np.where(y == 0)[0]], y[np.where(y == 0)[0]]
    x_1, y_1 = x[np.where(y == 1)[0]], y[np.where(y == 1)[0]]
    x_2, y_2 = x[np.where(y == 2)[0]], y[np.where(y == 2)[0]]
    acc_0 = accuracy_score(y_0, x_0)
    acc_1 = accuracy_score(y_1, x_1)
    acc_2 = accuracy_score(y_2, x_2)
    acc = accuracy_score(y, x)
    macro_f1 = f1_score(y, x, average='macro')
    return [acc_0, acc_1, acc_2, acc, macro_f1]
