import torch
import logging
import sklearn
import os
import socket
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, RMSprop, lr_scheduler, Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score

from models.neural_networks import *

SEED = 114
TIME = datetime.now().strftime('%b%d_%H-%M-%S')
BALANCED = True
DEVICE = 'cuda:0'
CLASS_WEIGHT = torch.tensor([1, 1, 1], dtype=torch.float)
LAYERS = [8, 16, 32, 64]
EPOCH = 30
LR = 2e-04
BATCH_SIZE = 1024
MODEL_DICT = None
PARALLEL = False
DECAY = True
DELTA_1 = 2
DELTA_2 = 2
ONE_HOT = False  # when using GRU and CNN, this should be false.


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=1.0, gamma=2, smooth=1e-04):
        # smooth not used..
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (alpha * ce_loss * ((1 - pt) ** gamma)).mean()

        return loss


def set_seed(digit: int):
    assert digit, 'Please specify a non-zero number when calling this func.'
    torch.manual_seed(digit)
    np.random.seed(digit)


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    x = logging.getLogger(__name__)
    x.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    x.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    x.addHandler(console)
    return x


def accuracy(x: np.ndarray, y: np.ndarray):
    # calculate averaged accuracy among input.
    # designed for this task only.
    x_0, y_0 = x[np.where(y == 0)[0]], y[np.where(y == 0)[0]]
    x_1, y_1 = x[np.where(y == 1)[0]], y[np.where(y == 1)[0]]
    x_2, y_2 = x[np.where(y == 2)[0]], y[np.where(y == 2)[0]]
    acc_0 = 0 if not len(y_0) or not len(x_0) else accuracy_score(y_0, x_0[..., 0], normalize=False)
    # acc_0 = 0 if np.isnan(acc_0) else acc_0
    acc_1 = 0 if not len(y_1) or not len(x_1) else accuracy_score(y_1, x_1[..., 0], normalize=False)
    # acc_1 = 0 if np.isnan(acc_1) else acc_1
    acc_2 = 0 if not len(y_2) or not len(x_2) else accuracy_score(y_2, x_2[..., 0], normalize=False)
    # acc_2 = 0 if np.isnan(acc_2) else acc_2
    return acc_0, acc_1, acc_2, len(np.where(y == 0)[0]), len(np.where(y == 1)[0]), len(np.where(y == 2)[0])


def _encode(char: float):
    if char == 0:
        return [1, 0, 0, 0]
    elif char == 1:
        return [0, 1, 0, 0]
    elif char == 2:
        return [0, 0, 1, 0]
    elif char == 3:
        return [0, 0, 0, 1]


def _eval_net(ml, loader, device):
    ml.eval()
    n_val = len(loader)  # the number of batch
    accs_0 = 0
    len_0 = 0
    accs_1 = 0
    len_1 = 0
    accs_2 = 0
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
            ac_0, ac_1, ac_2, l_0, l_1, l_2 = accuracy(refs.detach().cpu().numpy(), ys.detach().cpu().numpy())
            accs_0 += ac_0
            len_0 += l_0
            accs_1 += ac_1
            len_1 += l_1
            accs_2 += ac_2
            len_2 += l_2
            pbar.update()

    model.train()
    return accs_0 / len_0, accs_1 / len_1, accs_2 / len_2


class SequenceDataset(Dataset):
    def __init__(self, x: str, shuffle: bool = True, seed: int = 1, balanced=True):
        super(SequenceDataset, self).__init__()
        raw_data = np.load(x)
        self.features = raw_data['reads'].astype(np.float32)
        try:
            self.labels = raw_data['label']
        except KeyError:
            self.labels = np.ones(self.features.shape[0]) * -1
        self.labels = self.labels.astype(np.int64)
        self.ids = np.arange(len(self.labels)).astype(np.int64)
        if shuffle:
            self.features, self.labels, self.ids = sklearn.utils.shuffle(self.features,
                                                                         self.labels,
                                                                         self.ids, random_state=seed)
        if balanced:
            x_train_0, y_train_0 = self.features[np.where(self.labels == 0)[0]], self.labels[np.where(self.labels == 0)[0]]
            ids_0 = self.ids[np.where(self.labels == 0)[0]]
            x_train_1, y_train_1 = self.features[np.where(self.labels == 1)[0]], self.labels[np.where(self.labels == 1)[0]]
            ids_1 = self.ids[np.where(self.labels == 1)[0]]
            # randomly sample some data to make the dataset looks more equally distributed.
            x_train_1, y_train_1, ids_1 = x_train_1[:int(19713 * DELTA_1), :], y_train_1[:int(19713 * DELTA_1)], ids_1[:int(19713 * DELTA_1)]
            x_train_2, y_train_2 = self.features[np.where(self.labels == 2)[0]], self.labels[np.where(self.labels == 2)[0]]
            ids_2 = self.ids[np.where(self.labels == 2)[0]]
            # randomly sample some data to make the dataset looks more equally distributed.
            x_train_2, y_train_2, ids_2 = x_train_2[:int(19713 * DELTA_2), :], y_train_2[:int(19713 * DELTA_2)], ids_2[:int(19713 * DELTA_2)]
            self.features = np.concatenate([x_train_0, x_train_1, x_train_2], axis=0)
            self.labels = np.concatenate([y_train_0, y_train_1, y_train_2], axis=0)
            self.ids = np.concatenate([ids_0, ids_1, ids_2], axis=0)
            self.features, self.labels, self.ids = sklearn.utils.shuffle(self.features,
                                                                         self.labels,
                                                                         self.ids, random_state=seed)
        self.seed = seed

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature, label, index = self.features[idx], self.labels[idx], self.ids[idx]
        # feature = [_encode(i) for i in feature]
        return {'feature': torch.tensor(feature, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.int64),
                'index': torch.tensor(index, dtype=torch.int64)}


if __name__ == "__main__":
    # load dataset first
    if not ONE_HOT:
        train_data = SequenceDataset(x='./train_expand.npz', shuffle=True, seed=SEED, balanced=BALANCED)
        val_data = SequenceDataset(x='./val_expand.npz', shuffle=False, balanced=False)
        test_data = SequenceDataset(x='./test_expand.npz', shuffle=False, balanced=False)
    else:
        train_data = SequenceDataset(x='./train_one_hot.npz', shuffle=True, seed=SEED, balanced=BALANCED)
        val_data = SequenceDataset(x='./val_one_hot.npz', shuffle=False, balanced=False)
        test_data = SequenceDataset(x='./test_one_hot.npz', shuffle=False, balanced=False)
    # create logger and model path.
    if not os.path.exists('./runs'):
        os.makedirs('./runs')
    if SEED:
        set_seed(SEED)
    logger = create_logger('./runs/{}_{}.log'.format(TIME, socket.gethostname()))
    if 'cuda' in DEVICE:
        if not torch.cuda.is_available():
            raise ValueError('CUDA specified but not detected.')
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.info('%-40s %s\n' % ('Using device', DEVICE))
    # model = ANN(layers=LAYERS)
    # model = ResNet1D(in_channels=4 if ONE_HOT else 1,
    #                  layers=LAYERS,
    #                  n_class=3)
    model = GRU(embedding_dim=100, hidden_dim=128, num_layers=3)  # test
    if PARALLEL:
        model = nn.DataParallel(model)
    if MODEL_DICT:
        checkpoint = torch.load(MODEL_DICT)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=DEVICE)
    CLASS_WEIGHT = CLASS_WEIGHT.to(device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHT)
    # criterion = FocalLoss()
    # optimizer = Adam(model.parameters(), LR)
    optimizer = SGD(model.parameters(), LR, momentum=0.99)

    logger.info(' Config '.center(80, '-'))
    name_format = '%-40s %s\n' * 12
    logger.info(name_format % ("Learning Rate", LR,
                               "Decay", DECAY,
                               "Balanced Set", BALANCED,
                               "Loss Function", criterion,
                               "Class Weight", CLASS_WEIGHT.clone().detach().cpu().numpy(),
                               "Epoch", EPOCH,
                               "Batch Size", BATCH_SIZE,
                               'Seed', SEED,
                               "One-hot feature set", ONE_HOT,
                               "Training set size", train_data.__len__(),
                               "Validation Set Size", val_data.__len__(),
                               "Test Set Size", test_data.__len__()))
    logger.info(f'\t{model}')
    logger.info('-' * 80)
    if DECAY:
        scheduler = lr_scheduler.MultiStepLR(optimizer, [cc for cc in range(1, EPOCH, 1)], gamma=0.99)
    else:
        scheduler = None

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # already shuffled.
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    writer = SummaryWriter(log_dir='./runs/{}_{}/'.format(TIME, socket.gethostname()))
    global_step = 0
    # start training.
    for epoch in range(1, EPOCH + 1):
        model.train()
        with tqdm(total=train_data.__len__(), desc=f'Epoch {epoch}/{EPOCH}', unit='seqs') as pbar:
            for batch in train_loader:
                features = batch['feature'].to(device=DEVICE)
                if isinstance(model, GRU):
                    features = features.long()
                labels = batch['label'].to(device=DEVICE)

                with torch.no_grad():
                    if isinstance(model, ANN) or isinstance(model, GRU):
                        features = torch.squeeze(features)
                pred = model(features)
                loss = criterion(pred, labels)
                writer.add_scalar('loss/train_criterion', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(features.shape[0])
                global_step += 1

            if DECAY:
                scheduler.step()
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if not epoch % 1:
            mean_acc_0, mean_acc_1, mean_acc_2 = _eval_net(model, val_loader, DEVICE)
            logger.info('Validation Accuracy of class 0 for epoch {}: {}'.format(epoch, mean_acc_0))
            logger.info('Validation Accuracy of class 1 for epoch {}: {}'.format(epoch, mean_acc_1))
            logger.info('Validation Accuracy of class 2 for epoch {}: {}'.format(epoch, mean_acc_2))
            writer.add_scalar('val_acc/0', mean_acc_0, epoch)
            writer.add_scalar('val_acc/1', mean_acc_1, epoch)
            writer.add_scalar('val_acc/2', mean_acc_2, epoch)

    # training finished, start output evaluation results.
    logger.info("Saving testing set predictions for epoch {}".format(EPOCH))
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False)
    model.eval()
    outputs = []
    out_ids = []
    n_test = len(test_loader)  # the number of batch
    with tqdm(total=n_test, desc='Output Testing result', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            features, ids = batch['feature'].to(device=DEVICE), batch['index'].to(device='cpu')
            if isinstance(model, GRU):
                features = features.long()

            with torch.no_grad():
                if isinstance(model, ANN) or isinstance(model, GRU):
                    features = torch.squeeze(features)
                preds = model(features)
                preds = F.softmax(preds, dim=1)
                preds = torch.argmax(preds, dim=1, keepdim=True)
            outputs.append(preds.detach().cpu().numpy())
            out_ids.append(ids.detach().cpu().numpy())
            pbar.update()
    outputs = np.concatenate(outputs, axis=0)
    out_ids = np.concatenate(out_ids, axis=0)  # debug needed
    res = {'ID': out_ids, 'label': outputs[:, 0]}
    df = pd.DataFrame(data=res, dtype=np.int)
    df.to_csv('./runs/{}_{}/{}_epoch_{}.csv'.format(TIME, socket.gethostname(), model.__class__.__name__, EPOCH),
              index=False)
    logger.info("Saving model for final epoch {}".format(EPOCH))
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict()
    }, './runs/{}_{}/{}_epoch_{}.pt'.format(TIME, socket.gethostname(), model.__class__.__name__, EPOCH))
    writer.close()
