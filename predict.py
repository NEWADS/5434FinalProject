import torch
import os
import socket
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from utils.data import *
from utils.evaluation import *
from models.neural_networks import *

path = './runs/gru/Apr28_18-56-20_node4'
files = os.listdir(path)
config_path = os.path.join(path, files[np.where(['.npy' in file for file in files])[0][0]])
model_path = os.path.join(path, files[np.where(['.pt' in file for file in files])[0][0]])

config_dict = np.load(config_path, allow_pickle=True).item()

DEVICE = 'cuda:0'
TIME = datetime.now().strftime('%b%d_%H-%M-%S')
ONE_HOT = config_dict['one_hot']
# LAYERS = [8, 16, 32, 64]
BATCH_SIZE = config_dict['batch_size']
# MODEL_DICT = './runs/Apr17_21-24-48_node5/GRU_epoch_30.pt'


if __name__ == "__main__":
    # first, load the validation set.
    if ONE_HOT:
        path = './val_one_hot.npz'
    else:
        path = './val_expand.npz'
    val_data = SequenceDataset(x=path, shuffle=False, balanced=False)
    if 'cuda' in DEVICE:
        if not torch.cuda.is_available():
            raise ValueError('CUDA specified but not detected.')
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if config_dict['model'] == 'gru':
        model = GRU(config_dict['embedding_dim'], config_dict['hidden_dim'], config_dict['num_layers'], 3,
                    config_dict['drop_rate'])
    elif config_dict['model'] == 'resnet':
        model = ResNet1D(1 if not ONE_HOT else 3,
                         config_dict['layers'],
                         3,
                         config_dict['stages'])
    elif config_dict['model'] == 'resnet_m':
        model = ResNet1D_M(1 if not ONE_HOT else 3, config_dict['layers'], 3, config_dict['stages'],
                           config_dict['kernel_size'], config_dict['padding'], config_dict['drop_rate'])
    else:
        model = ANN(config_dict['layers'], 3, config_dict['drop_rate'])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=DEVICE)
    model.eval()
    if not os.path.exists('./results/{}'.format(model.__class__.__name__)):
        os.makedirs('./results/{}'.format(model.__class__.__name__))
    logger = create_logger('./results/{}/{}_{}.log'.format(model.__class__.__name__, TIME, socket.gethostname()))
    logger.info(' Config '.center(80, '-'))
    logger.info('%-40s %s\n' % ('Using device', DEVICE))
    logger.info('%-40s %s\n' % ('Evaluated Dataset', DEVICE))
    logger.info('%-40s %s\n' % ('Model Dict', path))
    logger.info('%-40s %s\n' % ('Batch Size', BATCH_SIZE))
    logger.info(f'\t{model}')
    logger.info('-' * 80)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    preds = []
    labels = []
    n_val = len(val_loader)  # the number of batch
    logger.info("Start evaluation...")
    with tqdm(total=n_val, desc='Evaluating', unit='batch', leave=False) as pbar:
        for batch in val_loader:
            xs, ys = batch['feature'], batch['label']
            if isinstance(model, GRU):
                xs = xs.long()
            xs = xs.to(device=DEVICE)
            ys = ys.to(device='cpu')
            with torch.no_grad():
                if isinstance(model, ANN) or isinstance(model, GRU):
                    xs = torch.squeeze(xs)
                refs = model(xs)
                refs = F.softmax(refs, dim=1)
                refs = torch.argmax(refs, dim=1, keepdim=True)
            preds.append(refs[..., 0].detach().cpu().numpy())
            labels.append(ys.detach().cpu().numpy())
            pbar.update()
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)  # debug needed
    results = evaluation_metrics(preds, labels)
    logger.info('%-40s %s\n' % ('Accuracy of Class 0', results[0]))
    logger.info('%-40s %s\n' % ('Accuracy of Class 1', results[1]))
    logger.info('%-40s %s\n' % ('Accuracy of Class 2', results[2]))
    logger.info('%-40s %s\n' % ('Mean Accuracy', results[3]))
    logger.info('%-40s %s\n' % ('Macro F1 Score', results[4]))
