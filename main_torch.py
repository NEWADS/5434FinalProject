import torch
import os
import socket
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from models.neural_networks import *
from utils.data import *
from utils.evaluation import *


config_dict = dict(
    seed=1,
    balanced=True,
    delta_1=2,
    delta_2=2,
    class_weight=[2, 1, 1],
    model='gru',
    # this entry works for ANN and CNN
    layers=[32, 64, 128, 256],
    # these entries work for CNN only
    stages=18,
    kernel_size=5,
    padding=2,
    # these entries work for GRU only
    embedding_dim=50,
    hidden_dim=256,
    num_layers=2,
    # this entry works for all networks
    drop_rate=0.2,
    epochs=210,
    # optimizer
    optim='sgd',
    lr=1e-03,
    decay=True,
    batch_size=1750,
    one_hot=False,
)

SEED = config_dict['seed']
TIME = datetime.now().strftime('%b%d_%H-%M-%S')
CLASS_WEIGHT = torch.tensor(config_dict['class_weight'], dtype=torch.float)
DEVICE = 'cuda:0'
BALANCED = config_dict['balanced']
# LAYERS = config_dict['layers']
EPOCH = config_dict['epochs']
LR = config_dict['lr']
BATCH_SIZE = config_dict['batch_size']
MODEL_DICT = None
PARALLEL = False
DECAY = config_dict['decay']
DELTA_1 = config_dict['delta_1']
DELTA_2 = config_dict['delta_2']
ONE_HOT = config_dict['one_hot']  # when using GRU and CNN, this should be false.


# def _encode(char: float):
#     if char == 0:
#         return [1, 0, 0, 0]
#     elif char == 1:
#         return [0, 1, 0, 0]
#     elif char == 2:
#         return [0, 0, 1, 0]
#     elif char == 3:
#         return [0, 0, 0, 1]


if __name__ == "__main__":
    # load dataset first
    if not ONE_HOT:
        train_data = SequenceDataset(x='./train_expand.npz', shuffle=True, seed=SEED, balanced=BALANCED,
                                     delta_1=DELTA_1, delta_2=DELTA_2)
        val_data = SequenceDataset(x='./val_expand.npz', shuffle=False, balanced=False)
        test_data = SequenceDataset(x='./test_expand.npz', shuffle=False, balanced=False)
    else:
        train_data = SequenceDataset(x='./train_one_hot.npz', shuffle=True, seed=SEED, balanced=BALANCED,
                                     delta_1=DELTA_1, delta_2=DELTA_2)
        val_data = SequenceDataset(x='./val_one_hot.npz', shuffle=False, balanced=False)
        test_data = SequenceDataset(x='./test_one_hot.npz', shuffle=False, balanced=False)
    # create logger and model path.
    if not os.path.exists('./runs/{}'.format(config_dict['model'])):
        os.makedirs('./runs/{}'.format(config_dict['model']))
    if SEED:
        set_seed(SEED)
    logger = create_logger('./runs/{}/{}_{}.log'.format(config_dict['model'], TIME, socket.gethostname()))
    if 'cuda' in DEVICE:
        if not torch.cuda.is_available():
            raise ValueError('CUDA specified but not detected.')
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.info('%-40s %s\n' % ('Using device', DEVICE))
    if config_dict['model'] == 'resnet_m':
        model = ResNet1D_M(in_channels=4 if ONE_HOT else 1, layers=config_dict['layers'], n_class=3,
                           stages=config_dict['stages'], kernel_size=config_dict['kernel_size'],
                           padding=config_dict['padding'], drop_rate=config_dict['drop_rate'])
    elif config_dict['model'] == 'ann':
        model = ANN(layers=config_dict['layers'], drop_rate=config_dict['drop_rate'])
    else:
        model = GRU(embedding_dim=config_dict['embedding_dim'], hidden_dim=config_dict['hidden_dim'],
                    num_layers=config_dict['num_layers'], drop_rate=config_dict['drop_rate'])
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
    if config_dict['optim'] == 'sgd':
        optimizer = SGD(model.parameters(), LR, momentum=0.99, weight_decay=5e-04)
    else:
        optimizer = Adam(model.parameters(), LR, weight_decay=5e-04)

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

    writer = SummaryWriter(log_dir='./runs/{}/{}_{}/'.format(config_dict['model'], TIME, socket.gethostname()))
    global_step = 0
    # start training.
    for epoch in range(1, EPOCH + 1):
        model.train()
        with tqdm(total=train_data.__len__(), desc=f'Epoch {epoch}/{EPOCH}', unit='seqs') as pbar:
            for batch in train_loader:
                features = batch['feature'].to(device=DEVICE)
                if isinstance(model, GRU) or config_dict['model'] == 'gru':
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
        if not epoch % 3 or epoch == EPOCH:
            mean_acc_0, mean_acc_1, mean_acc_2 = eval_net(model, val_loader, DEVICE)
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
    df.to_csv('./runs/{}/{}_{}/{}_epoch_{}.csv'.format(config_dict['model'], TIME, socket.gethostname(),
                                                       model.__class__.__name__, EPOCH),
              index=False)
    logger.info("Saving model for final epoch {}".format(EPOCH))
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict()
    }, './runs/{}/{}_{}/{}_epoch_{}.pt'.format(config_dict['model'], TIME, socket.gethostname(),
                                               model.__class__.__name__, EPOCH))
    np.save('./runs/{}/{}_{}/config_dict.npy'.format(config_dict['model'], TIME, socket.gethostname()), config_dict)
    writer.close()
